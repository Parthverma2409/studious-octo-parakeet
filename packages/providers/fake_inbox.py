"""In-memory MailProvider that replays fixtures from data/emails/."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from email import message_from_string
from pathlib import Path

from .mail import RawEmail


@dataclass
class FakeInbox:
    _queue: list[RawEmail] = field(default_factory=list)
    _seen: set[str] = field(default_factory=set)

    def deliver(self, email: RawEmail) -> None:
        self._queue.append(email)

    def deliver_fixture(self, path: Path, message_id: str | None = None) -> RawEmail:
        msg = message_from_string(path.read_text(encoding="utf-8"))
        body = msg.get_payload()
        if isinstance(body, list):
            body = "\n".join(str(p.get_payload()) for p in body)
        email = RawEmail(
            message_id=message_id or msg.get("Message-ID", path.stem),
            sender=msg.get("From", ""),
            subject=msg.get("Subject", ""),
            body=str(body),
            received_at=datetime.now(timezone.utc),
            headers=dict(msg.items()),
        )
        self.deliver(email)
        return email

    def fetch_new(self) -> list[RawEmail]:
        new = [e for e in self._queue if e.message_id not in self._seen]
        return new

    def mark_seen(self, message_id: str) -> None:
        self._seen.add(message_id)
