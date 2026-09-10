"""EmailIngestionModule -- polls a MailProvider and feeds the agent (FR2, UC2).

Marks a message seen only after the agent has finished with it, so a crash
mid-processing replays the message rather than losing it. Reconciliation is
idempotent (NFR2), which is what makes replaying safe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from packages.providers.mail import MailProvider
from packages.sync_agent.agent import AgentResult, SyncAgent


@dataclass
class EmailIngestionModule:
    mail: MailProvider
    agent: SyncAgent
    last_checked_at: datetime | None = None
    processed: list[str] = field(default_factory=list)

    def scan_inbox(self, now: datetime) -> list[AgentResult]:
        results: list[AgentResult] = []
        for email in self.mail.fetch_new():
            result = self.agent.handle_email(email, now)
            results.append(result)
            # Seen only on the way out: an exception above leaves the message
            # queued for the next scan instead of dropping a real booking change.
            self.mail.mark_seen(email.message_id)
            self.processed.append(email.message_id)
        self.last_checked_at = now
        return results
