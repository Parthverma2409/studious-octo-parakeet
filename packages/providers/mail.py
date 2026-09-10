"""MailProvider port."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class RawEmail:
    message_id: str
    sender: str
    subject: str
    body: str
    received_at: datetime
    headers: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class MailProvider(Protocol):
    def fetch_new(self) -> list[RawEmail]: ...
    def mark_seen(self, message_id: str) -> None: ...
