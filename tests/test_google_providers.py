"""Google adapters and the encrypted token store, driven by canned payloads.

There are no Google credentials on this machine, so nothing here touches the
network. Every test drives the adapters through the injected transport with
payloads shaped like real Calendar/Gmail v1 responses, and asserts on both
directions: the request that would have gone out, and the domain object built
from what came back.

That is a real limit and it is worth naming: these tests prove the adapter is
self-consistent and handles Google's documented error dialect. They cannot prove
the payload shapes are right. Only a live run can do that.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime, timedelta, timezone

import pytest

from packages.core.models import Alarm, BookingType, CalendarEvent
from packages.providers.calendar import CalendarProvider, RateLimitError
from packages.providers.gmail import GmailProvider
from packages.providers.google_calendar import GoogleCalendarProvider
from packages.providers.mail import MailProvider
from packages.providers.oauth import (
    SCOPES,
    GoogleApiClient,
    GoogleApiError,
    HttpResponse,
    OAuthCredentials,
    OAuthError,
    OAuthSession,
)
from packages.providers.tokens import EncryptedTokenStore, TokenStoreError, decrypt, encrypt

NOW = datetime(2026, 8, 18, 8, 0, tzinfo=timezone.utc)


# -- canned transport ------------------------------------------------------


class CannedTransport:
    """Replays scripted responses and records every request it was given.

    Recording the requests is the point: half of what an adapter can get wrong
    (a missing extended property, a PUT sent as a POST, a page token dropped)
    is invisible if you only assert on the parsed result.
    """

    def __init__(self, *responses: HttpResponse):
        self.responses = list(responses)
        self.calls: list[tuple[str, str, dict, bytes | None]] = []

    def __call__(self, method, url, headers, body):
        self.calls.append((method, url, headers, body))
        if not self.responses:
            raise AssertionError("transport called %d times with no response left" % len(self.calls))
        return self.responses.pop(0)

    @property
    def urls(self) -> list[str]:
        return [url for _, url, _, _ in self.calls]

    def payload(self, index: int) -> dict:
        body = self.calls[index][3]
        return json.loads(body.decode("utf-8")) if body else {}


def ok(payload: dict) -> HttpResponse:
    return HttpResponse(200, json.dumps(payload).encode("utf-8"))


def error(status: int, reason: str = "backendError") -> HttpResponse:
    body = {"error": {"code": status, "message": reason, "errors": [{"reason": reason}]}}
    return HttpResponse(status, json.dumps(body).encode("utf-8"))


def _client(*responses: HttpResponse, max_retries: int = 3) -> GoogleApiClient:
    """A client whose access token is live, so no test pays for a token refresh.

    The expiry is anchored to the real clock rather than to NOW: GoogleApiClient
    defaults `now` to wall-clock time, so a fixed 2026 expiry would silently make
    every one of these tests spend its first canned response on a refresh -- and
    would start doing so on a date that has nothing to do with the code.
    """
    credentials = OAuthCredentials(
        client_id="cid",
        client_secret="secret",
        refresh_token="refresh",
        access_token="access",
        expires_at=datetime.now(timezone.utc) + timedelta(days=365),
    )
    transport = CannedTransport(*responses)
    session = OAuthSession(credentials, transport=transport)
    # sleep is a no-op: backoff behaviour is asserted by counting calls, not by
    # making the suite wait for it.
    return GoogleApiClient(session, transport=transport, sleep=lambda _: None, max_retries=max_retries)


def _event_resource(event_id: str = "g1", summary: str = "Flight AI440 BLR to MAA", **extra) -> dict:
    resource = {
        "id": event_id,
        "status": "confirmed",
        "summary": summary,
        "start": {"dateTime": "2026-08-20T09:15:00Z"},
        "end": {"dateTime": "2026-08-20T11:45:00Z"},
        "extendedProperties": {"private": {"bookingRef": "AI440-XY12", "bookingType": "flight"}},
        "reminders": {"useDefault": False, "overrides": [{"method": "popup", "minutes": 60}]},
    }
    resource.update(extra)
    return resource


# -- the port ---------------------------------------------------------------


def test_google_adapters_satisfy_the_ports_they_claim() -> None:
    """The adapters must be substitutable for the fakes without a shim.

    NFR6 is only true if this holds; a Google adapter that needed SyncAgent
    changed to accommodate it would mean the port was decorative.
    """
    assert isinstance(GoogleCalendarProvider(_client()), CalendarProvider)
    assert isinstance(GmailProvider(_client()), MailProvider)


def test_scopes_are_least_privilege_and_gmail_is_read_only() -> None:
    """NFR3. Widening a scope must be a deliberate, visible edit."""
    assert SCOPES == (
        "https://www.googleapis.com/auth/calendar.events",
        "https://www.googleapis.com/auth/gmail.readonly",
    )
    # The bare "calendar" scope also grants calendar creation and ACL editing;
    # a gmail scope other than readonly grants sending or deleting mail.
    assert "https://www.googleapis.com/auth/calendar" not in SCOPES
    assert not any(s.startswith("https://www.googleapis.com/auth/gmail.") and not s.endswith("readonly") for s in SCOPES)


# -- calendar: reads --------------------------------------------------------


def test_list_events_maps_a_google_resource_onto_the_domain_model() -> None:
    """booking_ref and booking_type must survive the round trip.

    They live in extendedProperties because Google's resource has no field for
    them. Losing them does not fail loudly -- it makes FR3 matching miss, and
    every re-notification then looks like a brand new booking.
    """
    client = _client(ok({"items": [_event_resource()]}))
    events = GoogleCalendarProvider(client).list_events()

    assert len(events) == 1
    event = events[0]
    assert event.event_id == "g1"
    assert event.title == "Flight AI440 BLR to MAA"
    assert event.start_time == datetime(2026, 8, 20, 9, 15)
    assert event.end_time == datetime(2026, 8, 20, 11, 45)
    assert event.booking_ref == "AI440-XY12"
    assert event.booking_type is BookingType.FLIGHT
    assert event.alarm is not None
    assert event.alarm.trigger_time == datetime(2026, 8, 20, 8, 15)


def test_list_events_returns_naive_times_like_the_rest_of_the_domain() -> None:
    """Naive and aware datetimes raise TypeError on comparison.

    The domain model is naive throughout, so one aware datetime leaking out of
    this adapter would turn every reschedule comparison into a crash rather than
    a wrong answer -- and it would only appear once a real calendar was attached.
    """
    resource = _event_resource(start={"dateTime": "2026-08-20T14:45:00+05:30"}, end={"dateTime": "2026-08-20T16:00:00+05:30"})
    events = GoogleCalendarProvider(_client(ok({"items": [resource]}))).list_events()

    assert events[0].start_time.tzinfo is None
    # 14:45 +05:30 is 09:15 UTC.
    assert events[0].start_time == datetime(2026, 8, 20, 9, 15)


def test_list_events_skips_cancelled_resources() -> None:
    """Google keeps deleted events in the list as status=cancelled.

    Returning them would make a cancelled booking look present, so the agent
    would decide a cancellation had already been applied when it had not.
    """
    client = _client(ok({"items": [_event_resource("g1"), _event_resource("g2", status="cancelled")]}))
    assert [e.event_id for e in GoogleCalendarProvider(client).list_events()] == ["g1"]


def test_list_events_follows_every_page() -> None:
    """A calendar larger than one page must not silently truncate.

    A truncated list is the worst kind of wrong here: the missing event reads as
    "not booked", and the agent creates a duplicate.
    """
    page1 = ok({"items": [_event_resource("g1")], "nextPageToken": "TOKEN-2"})
    page2 = ok({"items": [_event_resource("g2", start={"dateTime": "2026-08-21T09:15:00Z"}, end={"dateTime": "2026-08-21T10:15:00Z"})]})
    client = _client(page1, page2)
    transport = client.transport

    events = GoogleCalendarProvider(client).list_events()

    assert [e.event_id for e in events] == ["g1", "g2"]
    assert "pageToken=TOKEN-2" in transport.urls[1]
    assert "pageToken" not in transport.urls[0]


def test_pagination_that_never_terminates_is_stopped() -> None:
    """A server repeating the same token would otherwise loop forever inside a sync."""
    looping = [ok({"items": [], "nextPageToken": "SAME"}) for _ in range(60)]
    client = _client(*looping)
    with pytest.raises(GoogleApiError, match="pagination did not terminate"):
        GoogleCalendarProvider(client).list_events()


# -- calendar: writes -------------------------------------------------------


def test_create_event_sends_the_booking_reference_and_the_alarm() -> None:
    """FR4: the alarm is part of the event, not an afterthought."""
    client = _client(ok(_event_resource()))
    transport = client.transport
    event = CalendarEvent(
        event_id="",
        title="Flight AI440 BLR to MAA",
        start_time=datetime(2026, 8, 20, 9, 15),
        booking_ref="AI440-XY12",
        booking_type=BookingType.FLIGHT,
        location="BLR",
    ).with_alarm_offset(60, "AL-1")

    created = GoogleCalendarProvider(client).create_event(event)

    method, url, headers, _ = transport.calls[0]
    body = transport.payload(0)
    assert method == "POST"
    assert url.endswith("/calendars/primary/events")
    assert headers["Authorization"] == "Bearer access"
    assert body["summary"] == "Flight AI440 BLR to MAA"
    assert body["start"] == {"dateTime": "2026-08-20T09:15:00", "timeZone": "UTC"}
    assert body["extendedProperties"]["private"] == {"bookingRef": "AI440-XY12", "bookingType": "flight"}
    assert body["reminders"]["overrides"] == [{"method": "popup", "minutes": 60}]
    assert body["location"] == "BLR"
    # The id comes back from Google, not from us.
    assert created.event_id == "g1"


def test_create_event_supplies_an_end_time_google_would_reject_without() -> None:
    """The domain allows end_time=None; the API does not."""
    client = _client(ok(_event_resource()))
    transport = client.transport
    GoogleCalendarProvider(client).create_event(
        CalendarEvent(event_id="", title="Dentist", start_time=datetime(2026, 8, 20, 9, 15))
    )
    assert transport.payload(0)["end"] == {"dateTime": "2026-08-20T10:15:00", "timeZone": "UTC"}


def test_update_event_puts_to_the_event_id() -> None:
    """A reschedule must address the existing event, never create a second one."""
    moved = _event_resource(start={"dateTime": "2026-08-20T14:00:00Z"}, end={"dateTime": "2026-08-20T15:00:00Z"})
    client = _client(ok(moved))
    transport = client.transport

    result = GoogleCalendarProvider(client).update_event(
        CalendarEvent(event_id="g1", title="Flight AI440 BLR to MAA", start_time=datetime(2026, 8, 20, 14, 0))
    )

    method, url, _, _ = transport.calls[0]
    assert method == "PUT"
    assert url.endswith("/calendars/primary/events/g1")
    assert result.start_time == datetime(2026, 8, 20, 14, 0)


def test_update_event_without_an_id_is_refused_before_any_request() -> None:
    """Without an id the API would happily create a duplicate event."""
    client = _client()
    with pytest.raises(ValueError, match="event_id"):
        GoogleCalendarProvider(client).update_event(
            CalendarEvent(event_id="", title="x", start_time=datetime(2026, 8, 20, 9, 15))
        )
    assert client.transport.calls == []


def test_delete_event_issues_a_delete_and_tolerates_an_empty_body() -> None:
    """Google answers 204 with no body; parsing it as JSON must not throw."""
    client = _client(HttpResponse(204, b""))
    transport = client.transport
    GoogleCalendarProvider(client).delete_event("g1")

    method, url, _, _ = transport.calls[0]
    assert method == "DELETE"
    assert url.endswith("/calendars/primary/events/g1")


def test_an_event_with_no_alarm_clears_the_calendar_default() -> None:
    """FR4: cancelling an alarm must actually cancel it.

    Omitting `reminders` leaves the calendar's default reminder in place, so a
    removed alarm would quietly keep firing.
    """
    client = _client(ok(_event_resource()))
    transport = client.transport
    GoogleCalendarProvider(client).create_event(
        CalendarEvent(event_id="", title="Dentist", start_time=datetime(2026, 8, 20, 9, 15))
    )
    assert transport.payload(0)["reminders"] == {"useDefault": False, "overrides": []}


def test_a_cancelled_alarm_is_not_sent_as_an_active_reminder() -> None:
    """Alarm.is_active False means the user turned it off."""
    client = _client(ok(_event_resource()))
    transport = client.transport
    event = CalendarEvent(
        event_id="",
        title="Dentist",
        start_time=datetime(2026, 8, 20, 9, 15),
        alarm=Alarm("AL-1", datetime(2026, 8, 20, 8, 15)).cancel(),
    )
    GoogleCalendarProvider(client).create_event(event)
    assert transport.payload(0)["reminders"]["overrides"] == []


def test_an_event_google_returns_without_a_start_fails_loudly() -> None:
    """Dropping it would read as 'not on the calendar' and cause a duplicate."""
    client = _client(ok({"items": [{"id": "g9", "summary": "broken", "status": "confirmed"}]}))
    with pytest.raises(GoogleApiError, match="no start time"):
        GoogleCalendarProvider(client).list_events()


# -- error mapping ----------------------------------------------------------


def test_429_becomes_the_existing_RateLimitError() -> None:
    """UC1 E3 is written against RateLimitError; a new error type would bypass it.

    The agent catches RateLimitError to queue the write and tell the user about
    the delay. A GoogleApiError here would escape that handler and surface as a
    crash instead of a queued change.
    """
    client = _client(error(429, "rateLimitExceeded"))
    with pytest.raises(RateLimitError):
        GoogleCalendarProvider(client).delete_event("g1")


def test_403_with_a_quota_reason_is_a_rate_limit() -> None:
    """Google overloads 403: for quota it is transient, for scopes it is not."""
    client = _client(error(403, "userRateLimitExceeded"))
    with pytest.raises(RateLimitError):
        GoogleCalendarProvider(client).delete_event("g1")


def test_403_for_insufficient_permission_is_not_treated_as_a_rate_limit() -> None:
    """A scope error must not be retried or queued: retrying never fixes it.

    Mapping every 403 to RateLimitError would turn a misconfigured scope into a
    change that appears queued and never lands -- silent, and indistinguishable
    from a busy API.
    """
    client = _client(error(403, "insufficientPermissions"))
    with pytest.raises(GoogleApiError, match="scopes") as raised:
        GoogleCalendarProvider(client).delete_event("g1")
    assert not isinstance(raised.value, RateLimitError)


def test_5xx_is_retried_with_bounded_backoff_and_then_succeeds() -> None:
    """A transient backend error must not surface as a failed booking change."""
    delays: list[float] = []
    client = _client(error(500, "backendError"), error(503, "backendError"), ok({"items": [_event_resource()]}))
    client.sleep = delays.append

    events = GoogleCalendarProvider(client).list_events()

    assert [e.event_id for e in events] == ["g1"]
    assert len(client.transport.calls) == 3
    # Bounded and growing: 0.5s then 1.0s, never unbounded.
    assert delays == [0.5, 1.0]


def test_5xx_that_never_recovers_gives_up_instead_of_retrying_forever() -> None:
    """An unbounded retry loop against a degraded API is a self-inflicted outage."""
    client = _client(*[error(503) for _ in range(10)], max_retries=2)
    client.sleep = lambda _: None
    with pytest.raises(GoogleApiError, match="after 2 retries"):
        GoogleCalendarProvider(client).list_events()
    assert len(client.transport.calls) == 3  # the original plus two retries


def test_an_error_message_never_quotes_the_query_string() -> None:
    """Error text ends up in tickets and logs; URLs are quoted without thinking."""
    client = _client(error(400, "badRequest"))
    with pytest.raises(GoogleApiError) as raised:
        GoogleCalendarProvider(client).list_events()
    assert "?" not in str(raised.value)
    assert "maxResults" not in str(raised.value)


# -- OAuth ------------------------------------------------------------------


def test_an_expired_access_token_is_refreshed_before_the_call() -> None:
    credentials = OAuthCredentials("cid", "secret", refresh_token="refresh", access_token="stale", expires_at=NOW - timedelta(minutes=5))
    transport = CannedTransport(
        ok({"access_token": "fresh", "expires_in": 3600}),
        ok({"items": []}),
    )
    session = OAuthSession(credentials, transport=transport)
    client = GoogleApiClient(session, transport=transport, sleep=lambda _: None)

    GoogleCalendarProvider(client).list_events()

    assert transport.calls[0][1].endswith("/token")
    assert transport.calls[1][2]["Authorization"] == "Bearer fresh"
    assert credentials.access_token == "fresh"


def test_a_refresh_never_overwrites_a_good_refresh_token_with_nothing() -> None:
    """Google re-issues a refresh token only on first consent.

    Clobbering it with the absent one in a refresh response would silently make
    the credential un-renewable, and the failure appears an hour later.
    """
    credentials = OAuthCredentials("cid", "secret", refresh_token="keepme", expires_at=None)
    transport = CannedTransport(ok({"access_token": "fresh", "expires_in": 3600}))
    OAuthSession(credentials, transport=transport).refresh(NOW)
    assert credentials.refresh_token == "keepme"


def test_a_failed_refresh_does_not_echo_the_response_body() -> None:
    """A token endpoint's body is the last place to be careless."""
    transport = CannedTransport(HttpResponse(400, json.dumps({"error": "invalid_grant", "id_token": "SENSITIVE"}).encode()))
    session = OAuthSession(OAuthCredentials("cid", "secret", refresh_token="r"), transport=transport)
    with pytest.raises(OAuthError) as raised:
        session.refresh(NOW)
    assert "invalid_grant" in str(raised.value)
    assert "SENSITIVE" not in str(raised.value)


def test_credentials_never_appear_in_their_own_repr() -> None:
    """NFR3. A dataclass repr is the most common way a secret reaches a log."""
    credentials = OAuthCredentials("cid", "SECRET-VALUE", refresh_token="REFRESH-VALUE", access_token="ACCESS-VALUE")
    text = repr(credentials)
    assert "cid" in text
    for secret in ("SECRET-VALUE", "REFRESH-VALUE", "ACCESS-VALUE"):
        assert secret not in text


def test_refresh_without_a_refresh_token_says_what_to_do() -> None:
    session = OAuthSession(OAuthCredentials("cid", "secret"), transport=CannedTransport())
    with pytest.raises(OAuthError, match="authorisation flow"):
        session.refresh(NOW)


# -- token store (NFR3) -----------------------------------------------------

PASSPHRASE = "correct horse battery staple"


def _payload() -> dict:
    return OAuthCredentials("cid", "s3cret", refresh_token="1//REFRESH-TOKEN", access_token="ya29.ACCESS").to_dict()


def test_token_store_round_trips_the_credential(tmp_path) -> None:
    store = EncryptedTokenStore(tmp_path / "token.json", PASSPHRASE)
    store.save(_payload())
    assert OAuthCredentials.from_dict(store.load()).refresh_token == "1//REFRESH-TOKEN"


def test_the_file_on_disk_contains_no_plaintext_token(tmp_path) -> None:
    """The whole point. A refresh token in the clear owns the account."""
    path = tmp_path / "token.json"
    EncryptedTokenStore(path, PASSPHRASE).save(_payload())
    raw = path.read_text(encoding="utf-8")

    assert "1//REFRESH-TOKEN" not in raw
    assert "ya29.ACCESS" not in raw
    assert "s3cret" not in raw
    # And not merely base64-of-plaintext, which would look encrypted and not be.
    assert base64.b64encode(b"1//REFRESH-TOKEN").decode("ascii") not in raw


def test_two_saves_of_the_same_payload_produce_different_ciphertext(tmp_path) -> None:
    """A fresh salt and nonce per save.

    Reusing them would repeat the keystream, and two files XORed together would
    leak the plaintext -- the classic way a stream cipher is misused.
    """
    first = encrypt(b"same", PASSPHRASE)
    second = encrypt(b"same", PASSPHRASE)
    assert first["ciphertext"] != second["ciphertext"]
    assert first["salt"] != second["salt"]
    assert first["nonce"] != second["nonce"]
    assert decrypt(first, PASSPHRASE) == decrypt(second, PASSPHRASE) == b"same"


def test_a_flipped_ciphertext_byte_is_detected_not_decrypted(tmp_path) -> None:
    """This is what 'authenticated' has to mean.

    A raw stream cipher decrypts tampered ciphertext into attacker-chosen
    plaintext without complaint: flip a bit in the ciphertext, flip the same bit
    in the token. Encrypt-then-MAC turns that into a clean refusal.
    """
    path = tmp_path / "token.json"
    store = EncryptedTokenStore(path, PASSPHRASE)
    store.save(_payload())

    envelope = json.loads(path.read_text(encoding="utf-8"))
    raw = bytearray(base64.b64decode(envelope["ciphertext"]))
    raw[0] ^= 0x01
    envelope["ciphertext"] = base64.b64encode(bytes(raw)).decode("ascii")
    path.write_text(json.dumps(envelope), encoding="utf-8")

    with pytest.raises(TokenStoreError, match="failed authentication"):
        store.load()


@pytest.mark.parametrize("field_name", ["salt", "nonce", "tag"])
def test_tampering_with_any_envelope_field_is_detected(tmp_path, field_name: str) -> None:
    """The MAC covers the whole envelope, not just the ciphertext.

    If it covered only the ciphertext, swapping in the salt from an older file
    would decrypt cleanly to different bytes.
    """
    path = tmp_path / "token.json"
    store = EncryptedTokenStore(path, PASSPHRASE)
    store.save(_payload())

    envelope = json.loads(path.read_text(encoding="utf-8"))
    raw = bytearray(base64.b64decode(envelope[field_name]))
    raw[0] ^= 0xFF
    envelope[field_name] = base64.b64encode(bytes(raw)).decode("ascii")
    path.write_text(json.dumps(envelope), encoding="utf-8")

    with pytest.raises(TokenStoreError):
        store.load()


def test_truncating_the_ciphertext_is_detected(tmp_path) -> None:
    """Truncation must not yield a short but 'valid' credential."""
    path = tmp_path / "token.json"
    store = EncryptedTokenStore(path, PASSPHRASE)
    store.save(_payload())

    envelope = json.loads(path.read_text(encoding="utf-8"))
    raw = base64.b64decode(envelope["ciphertext"])[:-8]
    envelope["ciphertext"] = base64.b64encode(raw).decode("ascii")
    path.write_text(json.dumps(envelope), encoding="utf-8")

    with pytest.raises(TokenStoreError):
        store.load()


def test_the_wrong_passphrase_fails_cleanly_rather_than_returning_garbage(tmp_path) -> None:
    """A stream cipher will happily 'decrypt' under the wrong key.

    Without the MAC this returns random bytes, which then fail to parse as JSON
    somewhere far away from the actual cause -- or worse, parse.
    """
    path = tmp_path / "token.json"
    EncryptedTokenStore(path, PASSPHRASE).save(_payload())

    with pytest.raises(TokenStoreError) as raised:
        EncryptedTokenStore(path, "not the passphrase").load()
    # One honest message: a failed MAC genuinely cannot tell the two causes apart.
    assert "wrong passphrase" in str(raised.value)
    assert "modified" in str(raised.value)


def test_a_missing_token_file_says_what_to_do(tmp_path) -> None:
    with pytest.raises(TokenStoreError, match="authorisation flow"):
        EncryptedTokenStore(tmp_path / "absent.json", PASSPHRASE).load()


def test_an_unknown_format_version_is_refused_rather_than_guessed(tmp_path) -> None:
    """If the construction ever changes, an old file must fail loudly."""
    envelope = encrypt(b"{}", PASSPHRASE)
    envelope["version"] = "bsync-token-v0"
    with pytest.raises(TokenStoreError, match="unsupported token file version"):
        decrypt(envelope, PASSPHRASE)


def test_lowering_the_kdf_iterations_in_the_file_is_refused(tmp_path) -> None:
    """Otherwise an attacker edits the file to make cracking cheap and we comply."""
    envelope = encrypt(b"{}", PASSPHRASE)
    envelope["kdf"]["iterations"] = 1
    with pytest.raises(TokenStoreError, match="refusing to guess"):
        decrypt(envelope, PASSPHRASE)


def test_the_store_does_not_leak_the_passphrase_in_its_repr(tmp_path) -> None:
    store = EncryptedTokenStore(tmp_path / "token.json", "PASSPHRASE-VALUE")
    assert "PASSPHRASE-VALUE" not in repr(store)


def test_a_refreshed_token_is_written_back_encrypted(tmp_path) -> None:
    """The refresh path must not be the one that writes a token in the clear."""
    path = tmp_path / "token.json"
    store = EncryptedTokenStore(path, PASSPHRASE)
    credentials = OAuthCredentials("cid", "secret", refresh_token="1//REFRESH-TOKEN")
    transport = CannedTransport(ok({"access_token": "ya29.FRESH", "expires_in": 3600}))

    OAuthSession(credentials, transport=transport, store=store).refresh(NOW)

    assert "ya29.FRESH" not in path.read_text(encoding="utf-8")
    assert store.load()["access_token"] == "ya29.FRESH"


# -- gmail ------------------------------------------------------------------


def _gmail_message(message_id: str = "m1", body: str = "Your booking AI440 is cancelled.") -> dict:
    encoded = base64.urlsafe_b64encode(body.encode("utf-8")).decode("ascii").rstrip("=")
    return {
        "id": message_id,
        "threadId": "t1",
        "internalDate": "1787040000000",
        "snippet": "Your booking...",
        "payload": {
            "mimeType": "multipart/alternative",
            "headers": [
                {"name": "From", "value": "noreply@airline.example"},
                {"name": "Subject", "value": "Cancellation: AI440"},
                {"name": "Message-ID", "value": "<abc@airline.example>"},
            ],
            "parts": [
                {"mimeType": "text/html", "body": {"data": base64.urlsafe_b64encode(b"<p>markup</p>").decode().rstrip("=")}},
                {"mimeType": "text/plain", "body": {"data": encoded}},
            ],
        },
    }


def test_fetch_new_builds_a_RawEmail_from_a_gmail_message() -> None:
    client = _client(ok({"messages": [{"id": "m1"}]}), ok(_gmail_message()))
    emails = GmailProvider(client).fetch_new()

    assert len(emails) == 1
    email = emails[0]
    assert email.message_id == "m1"
    assert email.sender == "noreply@airline.example"
    assert email.subject == "Cancellation: AI440"
    assert "AI440 is cancelled" in email.body
    assert email.headers["Message-ID"] == "<abc@airline.example>"
    assert email.received_at.tzinfo is None


def test_the_plain_text_part_is_preferred_over_the_html_twin() -> None:
    """email_extract.py is written against plain text.

    Feeding it markup does not fail -- it produces confident nonsense, which is
    the failure mode UC2 E1 exists to catch and which is better avoided here.
    """
    client = _client(ok({"messages": [{"id": "m1"}]}), ok(_gmail_message()))
    assert "<p>" not in GmailProvider(client).fetch_new()[0].body


def test_gmail_ids_not_sender_supplied_message_ids_are_the_idempotency_key() -> None:
    """NFR2 depends on the same notification producing the same key twice.

    A sender controls the RFC822 Message-ID and can omit or repeat it; Gmail's
    own id it cannot touch.
    """
    client = _client(ok({"messages": [{"id": "m1"}]}), ok(_gmail_message()))
    email = GmailProvider(client).fetch_new()[0]
    assert email.message_id == "m1"
    assert email.message_id != email.headers["Message-ID"]


def test_a_message_body_with_missing_base64_padding_still_decodes() -> None:
    """Gmail uses unpadded base64url; strict b64decode raises on it.

    Roughly three messages in four have a length that needs padding restored, so
    getting this wrong fails most of the time, not rarely.
    """
    client = _client(ok({"messages": [{"id": "m1"}]}), ok(_gmail_message(body="abcde")))
    assert GmailProvider(client).fetch_new()[0].body == "abcde"


def test_fetch_new_pages_through_the_message_list() -> None:
    client = _client(
        ok({"messages": [{"id": "m1"}], "nextPageToken": "P2"}),
        ok({"messages": [{"id": "m2"}]}),
        ok(_gmail_message("m1")),
        ok(_gmail_message("m2")),
    )
    assert [e.message_id for e in GmailProvider(client).fetch_new()] == ["m1", "m2"]


def test_mark_seen_suppresses_a_message_on_the_next_scan() -> None:
    """The read state is local because the scope is read-only, and it still works."""
    client = _client(
        ok({"messages": [{"id": "m1"}]}),
        ok(_gmail_message()),
        ok({"messages": [{"id": "m1"}]}),
    )
    provider = GmailProvider(client)
    assert len(provider.fetch_new()) == 1
    provider.mark_seen("m1")
    assert provider.fetch_new() == []
    # Three calls, not four: the second scan never fetched the seen message.
    assert len(client.transport.calls) == 3


def test_mark_seen_issues_no_write_to_gmail() -> None:
    """It cannot: gmail.readonly is the granted scope.

    Asserting it here means a future 'let's just mark it read' change has to
    confront the scope decision instead of quietly widening it.
    """
    client = _client()
    GmailProvider(client).mark_seen("m1")
    assert client.transport.calls == []


def test_gmail_rate_limits_use_the_same_error_vocabulary() -> None:
    client = _client(error(429, "rateLimitExceeded"))
    with pytest.raises(RateLimitError):
        GmailProvider(client).fetch_new()
