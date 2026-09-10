"""GmailProvider -- the real MailProvider behind the same port (FR2).

Drop-in for ``FakeInbox``: ``EmailIngestionModule`` and ``SyncAgent`` are not
touched to accommodate it. What is worth knowing about this adapter:

  * **It is read-only, on purpose.** The granted scope is
    ``gmail.readonly`` (see ``oauth.SCOPES``), so ``mark_seen`` *cannot* write
    the read state back to the server -- that needs ``gmail.modify``, which also
    grants label editing, trashing and archiving. A booking agent that can
    silently trash mail is a much worse outcome than one that keeps its own
    seen-set, so the seen-set is kept locally and the wider scope is not
    requested. The cost is stated openly on ``mark_seen``.
  * **Gmail's message id is the idempotency key**, not the RFC822
    ``Message-ID`` header. NFR2 depends on the same notification producing the
    same key twice; a sender that omits or duplicates ``Message-ID`` would break
    that, and Gmail's own id cannot be influenced by the sender.

Not exercised against Gmail here -- no credentials on this machine. Every call
goes through an injected ``GoogleApiClient`` fed with recorded payloads.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .mail import RawEmail
from .oauth import GoogleApiClient

API_ROOT = "https://gmail.googleapis.com/gmail/v1/users/me"

#: Gmail's page maximum for messages.list.
PAGE_SIZE = 100

#: Default server-side filter. Narrowing the query at the server is both a
#: quota saving and a privacy measure: mail the agent never fetches is mail it
#: cannot mishandle.
DEFAULT_QUERY = "in:inbox newer_than:30d"


@dataclass
class GmailProvider:
    """Implements ``MailProvider`` against Gmail API v1."""

    client: GoogleApiClient
    query: str = DEFAULT_QUERY
    max_messages: int = 50
    #: Locally tracked read state. See mark_seen for why this is not server-side.
    _seen: set[str] = field(default_factory=set)

    # -- MailProvider ------------------------------------------------------

    def fetch_new(self) -> list[RawEmail]:
        """Every unseen message matching the query, oldest first.

        Oldest first matters: UC5 (out-of-order arrival) is about the agent
        handling a reschedule that lands before its confirmation, and feeding
        the queue newest-first would manufacture that condition on every scan.
        """
        listed = self.client.paginate(
            "%s/messages" % API_ROOT,
            params={"q": self.query, "maxResults": PAGE_SIZE},
            items_key="messages",
        )
        emails: list[RawEmail] = []
        for stub in listed[: self.max_messages]:
            message_id = stub.get("id")
            if not message_id or message_id in self._seen:
                continue
            emails.append(self._fetch_one(message_id))
        return sorted(emails, key=lambda e: (e.received_at, e.message_id))

    def mark_seen(self, message_id: str) -> None:
        """Record locally that this message has been processed.

        This does NOT mark the message read in Gmail. Doing so requires the
        ``gmail.modify`` scope, which also grants trashing and label editing --
        far more authority than "note that I have read this" is worth. The
        consequence is honest and bounded: the seen-set lives for the life of the
        process, so a restart re-reads recent mail. Reconciliation is idempotent
        (NFR2), which is exactly why re-reading is safe and losing a message is
        not; if durable read state is ever needed, the caller should persist this
        set rather than the scope being widened.
        """
        self._seen.add(message_id)

    # -- mapping -----------------------------------------------------------

    def _fetch_one(self, message_id: str) -> RawEmail:
        message = self.client.request(
            "GET", "%s/messages/%s" % (API_ROOT, message_id), params={"format": "full"}
        )
        payload = message.get("payload") or {}
        headers = {h.get("name", ""): h.get("value", "") for h in payload.get("headers") or []}
        return RawEmail(
            message_id=message.get("id", message_id),
            sender=headers.get("From", ""),
            subject=headers.get("Subject", ""),
            body=_body_text(payload) or message.get("snippet", ""),
            received_at=_received_at(message),
            headers=headers,
        )


def _received_at(message: dict) -> datetime:
    """``internalDate`` is epoch milliseconds, and is the arrival time Gmail
    itself sorts by -- more trustworthy than a ``Date:`` header the sender wrote.

    Returned naive-UTC to match the rest of the domain model, which never carries
    tzinfo; a mixed naive/aware datetime raises on comparison rather than
    misbehaving quietly, so the conversion happens here, once.
    """
    raw = message.get("internalDate")
    if raw is None:
        return datetime.now(timezone.utc).replace(tzinfo=None)
    return datetime.fromtimestamp(int(raw) / 1000.0, tz=timezone.utc).replace(tzinfo=None)


def _body_text(payload: dict) -> str:
    """Depth-first search for the plain-text part.

    Real booking mail is ``multipart/alternative`` with an HTML twin, and the
    extractor in ``packages/nlp/email_extract.py`` is written against plain text.
    Falling back to HTML would feed it markup and produce confident nonsense, so
    text/plain is preferred and HTML is used only when there is nothing else.
    """
    plain = _find_part(payload, "text/plain")
    if plain is not None:
        return _decode(plain)
    html = _find_part(payload, "text/html")
    if html is not None:
        return _decode(html)
    return _decode(payload)


def _find_part(node: dict, mime_type: str) -> dict | None:
    if node.get("mimeType") == mime_type and (node.get("body") or {}).get("data"):
        return node
    for part in node.get("parts") or []:
        found = _find_part(part, mime_type)
        if found is not None:
            return found
    return None


def _decode(node: dict) -> str:
    data = (node.get("body") or {}).get("data")
    if not data:
        return ""
    # Gmail uses base64url without padding; b64decode is strict about padding, so
    # it is restored here rather than being left to raise on every fourth message.
    padded = data + "=" * (-len(data) % 4)
    try:
        return base64.urlsafe_b64decode(padded).decode("utf-8", errors="replace")
    except (ValueError, TypeError):
        return ""
