"""OAuth2 for Google, plus the one authenticated HTTP client both adapters share.

Three things live here because they are one conversation with one server: the
scopes we ask for, the token refresh dance, and the retry / error mapping. Split
across ``google_calendar.py`` and ``gmail.py`` they would drift, and the failures
drift produces are the expensive ones -- a 403 that is really a quota error being
retried forever, or an access token reaching a log line.

Nothing in this module can be exercised against Google on this machine: there are
no credentials here. So every byte on the wire goes through an injected
transport. Production passes ``UrllibTransport``; tests pass a canned callable
and assert on the request that would have been sent.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Protocol, runtime_checkable

from .calendar import CalendarError, RateLimitError

#: Least-privilege scopes. Each one is the narrowest scope that still lets the
#: agent do its job; anything wider is a standing invitation to a bug.
#:
#:   calendar.events  -- FR4 needs to create, move and delete *events*. The wider
#:                       ".../auth/calendar" scope would also grant calendar
#:                       creation/deletion and ACL editing, which the agent never
#:                       does, so asking for it would be asking for damage it
#:                       cannot cause today but could cause after a bad refactor.
#:   gmail.readonly   -- FR2 only ever *reads* booking confirmations. The agent
#:                       never sends, replies, deletes or labels, so readonly is
#:                       the ceiling. This has a visible cost: see
#:                       GmailProvider.mark_seen, which cannot write the read
#:                       state back to the server. That cost is worth paying.
SCOPES: tuple[str, ...] = (
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/gmail.readonly",
)

AUTH_URI = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URI = "https://oauth2.googleapis.com/token"

#: Refresh this long before the token actually dies, so a slow request does not
#: start valid and finish expired.
EXPIRY_SKEW = timedelta(seconds=60)

#: Bounded exponential backoff for 5xx. Bounded matters: an unbounded retry loop
#: against a degraded API is a self-inflicted outage.
BACKOFF_BASE_SEC = 0.5
BACKOFF_MAX_SEC = 8.0
DEFAULT_MAX_RETRIES = 3

#: 403 is overloaded by Google: it means both "quota exhausted" and "you do not
#: have permission". Only these reasons are retry-worthy rate limits; anything
#: else that arrives as a 403 is a configuration error and must surface as one,
#: because retrying a permission failure never succeeds.
RATE_LIMIT_REASONS = frozenset(
    {"ratelimitexceeded", "userratelimitexceeded", "quotaexceeded", "dailylimitexceeded"}
)


class GoogleApiError(CalendarError):
    """A Google API call failed in a way retrying will not fix.

    It subclasses ``CalendarError`` so that code written against the
    ``CalendarProvider`` port -- which knows only ``CalendarError`` and
    ``RateLimitError`` -- keeps working when a Google adapter is swapped in for
    the fake. Gmail failures land on the same type, which is a naming wart
    accepted deliberately in exchange for one error vocabulary at the call site.
    """


class OAuthError(GoogleApiError):
    """Token exchange or refresh failed."""


@dataclass(frozen=True)
class HttpResponse:
    """A transport's answer. Never raises for a non-2xx status.

    HTTP errors are values here, not exceptions, because the retry and
    rate-limit policy needs to *inspect* a 429 or a 503 rather than catch it.
    """

    status: int
    body: bytes
    headers: dict[str, str] = field(default_factory=dict)

    def json(self) -> dict:
        """Parse the body, or return {} -- Google sends an empty 204 on delete."""
        if not self.body:
            return {}
        try:
            parsed = json.loads(self.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}


@runtime_checkable
class HttpTransport(Protocol):
    """One call, one response. The seam that makes this package testable."""

    def __call__(
        self, method: str, url: str, headers: dict[str, str], body: bytes | None
    ) -> HttpResponse: ...


class UrllibTransport:
    """stdlib-only transport. ``requests`` is not available and not needed."""

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout

    def __call__(
        self, method: str, url: str, headers: dict[str, str], body: bytes | None
    ) -> HttpResponse:
        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return HttpResponse(response.status, response.read(), dict(response.headers))
        except urllib.error.HTTPError as exc:
            # A 4xx/5xx is data the caller must classify, not an exception it
            # must guess at, so it comes back through the same channel.
            return HttpResponse(exc.code, exc.read(), dict(exc.headers or {}))
        except urllib.error.URLError as exc:
            raise GoogleApiError("network error talking to Google: %s" % exc.reason) from exc


@dataclass
class OAuthCredentials:
    """Client identity plus the tokens for one traveller.

    Every secret field is ``repr=False``. A dataclass repr is the most common way
    a credential ends up in a log, a traceback or a pytest assertion message, and
    there is no version of that which is acceptable.
    """

    client_id: str
    client_secret: str = field(repr=False)
    refresh_token: str = field(default="", repr=False)
    access_token: str = field(default="", repr=False)
    expires_at: datetime | None = None
    scopes: tuple[str, ...] = SCOPES

    def is_expired(self, now: datetime, skew: timedelta = EXPIRY_SKEW) -> bool:
        if not self.access_token:
            return True
        if self.expires_at is None:
            # No expiry recorded means we cannot prove it is live. Treat unknown
            # as expired: a needless refresh is cheap, a 401 mid-write is not.
            return True
        return now + skew >= self.expires_at

    def to_dict(self) -> dict:
        """Serialisable form for the encrypted store. Includes the secrets."""
        return {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "refresh_token": self.refresh_token,
            "access_token": self.access_token,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "scopes": list(self.scopes),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "OAuthCredentials":
        expires_at = payload.get("expires_at")
        return cls(
            client_id=payload.get("client_id", ""),
            client_secret=payload.get("client_secret", ""),
            refresh_token=payload.get("refresh_token", ""),
            access_token=payload.get("access_token", ""),
            expires_at=datetime.fromisoformat(expires_at) if expires_at else None,
            scopes=tuple(payload.get("scopes") or SCOPES),
        )


def authorization_url(client_id: str, redirect_uri: str, scopes: tuple[str, ...] = SCOPES) -> str:
    """Step 1 of the installed-app flow: the URL the traveller opens once.

    ``access_type=offline`` is what makes Google issue a refresh token, and
    ``prompt=consent`` forces it to be re-issued on re-authorisation; without
    both, a re-auth silently yields credentials that die in an hour.
    """
    query = urllib.parse.urlencode(
        {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": " ".join(scopes),
            "access_type": "offline",
            "prompt": "consent",
        }
    )
    return "%s?%s" % (AUTH_URI, query)


class OAuthSession:
    """Holds credentials and hands out a live Authorization header.

    Refresh happens here and nowhere else, so there is exactly one place that
    knows how to turn a refresh token into an access token, and exactly one place
    that writes the result back to the token store.
    """

    def __init__(
        self,
        credentials: OAuthCredentials,
        transport: HttpTransport | None = None,
        store=None,
    ):
        self.credentials = credentials
        self.transport = transport or UrllibTransport()
        #: Optional EncryptedTokenStore. A refreshed access token is persisted so
        #: a restart does not burn a refresh round-trip -- and so the file on disk
        #: never holds a token in the clear.
        self.store = store

    def authorization_header(self, now: datetime | None = None) -> dict[str, str]:
        now = now or datetime.now(timezone.utc)
        if self.credentials.is_expired(now):
            self.refresh(now)
        return {"Authorization": "Bearer %s" % self.credentials.access_token}

    def refresh(self, now: datetime | None = None) -> None:
        now = now or datetime.now(timezone.utc)
        if not self.credentials.refresh_token:
            raise OAuthError(
                "no refresh token available; re-run the authorisation flow "
                "(see authorization_url / exchange_code)"
            )
        body = urllib.parse.urlencode(
            {
                "client_id": self.credentials.client_id,
                "client_secret": self.credentials.client_secret,
                "refresh_token": self.credentials.refresh_token,
                "grant_type": "refresh_token",
            }
        ).encode("utf-8")
        response = self.transport(
            "POST",
            TOKEN_URI,
            {"Content-Type": "application/x-www-form-urlencoded"},
            body,
        )
        self._absorb(response, now)

    def exchange_code(self, code: str, redirect_uri: str, now: datetime | None = None) -> None:
        """Step 2 of the installed-app flow: authorisation code -> tokens."""
        now = now or datetime.now(timezone.utc)
        body = urllib.parse.urlencode(
            {
                "client_id": self.credentials.client_id,
                "client_secret": self.credentials.client_secret,
                "code": code,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            }
        ).encode("utf-8")
        response = self.transport(
            "POST",
            TOKEN_URI,
            {"Content-Type": "application/x-www-form-urlencoded"},
            body,
        )
        self._absorb(response, now)

    def _absorb(self, response: HttpResponse, now: datetime) -> None:
        payload = response.json()
        if response.status != 200 or "access_token" not in payload:
            # Only the machine-readable error code is quoted. The raw body is not
            # logged because a token endpoint's response body is the last place
            # you want to be careless about.
            raise OAuthError(
                "token endpoint returned %d (%s)"
                % (response.status, payload.get("error") or "no access_token in response")
            )
        self.credentials.access_token = payload["access_token"]
        if payload.get("refresh_token"):
            # Google re-issues one only on first consent; never overwrite a good
            # refresh token with an absent one.
            self.credentials.refresh_token = payload["refresh_token"]
        expires_in = payload.get("expires_in")
        self.credentials.expires_at = (
            now + timedelta(seconds=int(expires_in)) if expires_in else None
        )
        if self.store is not None:
            self.store.save(self.credentials.to_dict())


class GoogleApiClient:
    """Authenticated JSON calls with retry, backoff and Google's error dialect.

    Retry policy, stated once so both adapters cannot disagree:
      * 5xx        -- transient. Retried up to ``max_retries`` with bounded
                      exponential backoff, then raised as GoogleApiError.
      * 429        -- RateLimitError immediately, *not* retried here. UC1 E3
                      says a rate-limited write is queued and the user is told
                      about the delay; sleeping inside the request would turn a
                      queueable condition into a hung interaction.
      * 403        -- RateLimitError only when the body names a quota reason;
                      otherwise a permission error, which no amount of retrying
                      fixes, so it is raised at once.
    """

    def __init__(
        self,
        session: OAuthSession,
        transport: HttpTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        self.session = session
        self.transport = transport or session.transport
        self.sleep = sleep
        self.max_retries = max_retries

    def request(
        self,
        method: str,
        url: str,
        params: dict | None = None,
        payload: dict | None = None,
        now: datetime | None = None,
    ) -> dict:
        if params:
            url = "%s?%s" % (url, urllib.parse.urlencode(_clean(params)))
        body = json.dumps(payload).encode("utf-8") if payload is not None else None

        attempt = 0
        while True:
            headers = dict(self.session.authorization_header(now))
            if body is not None:
                headers["Content-Type"] = "application/json"
            response = self.transport(method, url, headers, body)

            if 200 <= response.status < 300:
                return response.json()

            self._raise_for_permanent(response, url)

            if attempt >= self.max_retries:
                raise GoogleApiError(
                    "%s %s failed with %d after %d retries: %s"
                    % (method, _redact(url), response.status, attempt, _reason(response))
                )
            self.sleep(min(BACKOFF_BASE_SEC * (2**attempt), BACKOFF_MAX_SEC))
            attempt += 1

    def paginate(
        self,
        url: str,
        params: dict | None = None,
        items_key: str = "items",
        max_pages: int = 50,
    ) -> list[dict]:
        """Follow ``nextPageToken`` to the end of a collection.

        ``max_pages`` is a guard, not a preference: a server that keeps handing
        back the same token would otherwise loop forever inside a sync.
        """
        collected: list[dict] = []
        page_params = dict(params or {})
        for _ in range(max_pages):
            page = self.request("GET", url, params=page_params)
            collected.extend(page.get(items_key) or [])
            token = page.get("nextPageToken")
            if not token:
                return collected
            page_params["pageToken"] = token
        raise GoogleApiError("pagination did not terminate after %d pages of %s" % (max_pages, _redact(url)))

    def _raise_for_permanent(self, response: HttpResponse, url: str) -> None:
        """Classify a non-2xx. Returns only when the status is worth retrying."""
        if response.status == 429:
            raise RateLimitError("Google API rate limit exceeded (429)")
        if response.status == 403:
            if _reason(response).lower() in RATE_LIMIT_REASONS:
                raise RateLimitError("Google API quota exceeded (403 %s)" % _reason(response))
            raise GoogleApiError(
                "403 from %s: %s -- check the granted scopes" % (_redact(url), _reason(response))
            )
        if response.status == 401:
            raise OAuthError("401 from %s: the access token was rejected" % _redact(url))
        if 400 <= response.status < 500:
            raise GoogleApiError("%d from %s: %s" % (response.status, _redact(url), _reason(response)))


def _reason(response: HttpResponse) -> str:
    """Google's machine-readable error reason, or its message, or the status."""
    error = response.json().get("error")
    if isinstance(error, dict):
        errors = error.get("errors") or []
        if errors and isinstance(errors[0], dict) and errors[0].get("reason"):
            return str(errors[0]["reason"])
        if error.get("status"):
            return str(error["status"])
        if error.get("message"):
            return str(error["message"])
    if isinstance(error, str):
        return error
    return "http %d" % response.status


def _redact(url: str) -> str:
    """Drop the query string before a URL reaches a message.

    Google never puts a token in a query string in this client, but error text
    has a way of ending up in tickets, and a URL is the easiest thing to quote
    without thinking.
    """
    return url.split("?", 1)[0]


def _clean(params: dict) -> dict:
    return {k: v for k, v in params.items() if v is not None}
