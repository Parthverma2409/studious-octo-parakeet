"""The bounded queue that stands between a live agent and a dead ingest service.

An unbounded in-process queue does not survive an outage, it converts one: the
ingest service goes down, trajectories pile up, and the agent process is killed
by the OOM reaper. The user loses their assistant because the *evaluation*
platform was unavailable, which is exactly backwards.

So the buffer has a ceiling and drops the OLDEST trajectory when it is reached.
Recent behaviour is what the platform is looking at, and a counted drop is a
known gap in the record; an unbounded queue is an unknown one.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Any


class BoundedBuffer:
    """Thread-safe, drop-oldest, with a counter for what it dropped.

    Both ends drop the oldest item: `put` evicts the head to make room for new
    work, and `put_front` (used when a send failed and the item goes back for
    retry) refuses the item when full, because at that moment the item being
    returned *is* the oldest one.
    """

    def __init__(self, capacity: int = 512):
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.capacity = capacity
        self._items: deque[Any] = deque()
        self._dropped = 0
        # One condition guards the deque, the drop counter and the stop flag, so
        # a worker blocked on an empty buffer is woken by either new work or
        # shutdown without a second synchronisation primitive to get wrong.
        self._cond = threading.Condition()
        self._closed = False

    def put(self, item: Any) -> bool:
        """Appends. Returns False if an older item had to be dropped to fit."""
        with self._cond:
            if self._closed:
                return False
            evicted = False
            while len(self._items) >= self.capacity:
                self._items.popleft()
                self._dropped += 1
                evicted = True
            self._items.append(item)
            self._cond.notify()
            return not evicted

    def put_front(self, item: Any) -> bool:
        """Returns an item to the head for retry. False if it had to be dropped."""
        with self._cond:
            if self._closed or len(self._items) >= self.capacity:
                self._dropped += 1
                return False
            self._items.appendleft(item)
            self._cond.notify()
            return True

    def take(self, timeout: float | None = None) -> Any | None:
        """Blocks for the oldest item. None on timeout or after `close`."""
        with self._cond:
            if not self._items and not self._closed:
                self._cond.wait(timeout)
            if self._items:
                return self._items.popleft()
            return None

    def close(self) -> None:
        """Wakes every waiter and refuses further writes. Idempotent."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def drain(self) -> list[Any]:
        with self._cond:
            items = list(self._items)
            self._items.clear()
            return items

    @property
    def dropped(self) -> int:
        with self._cond:
            return self._dropped

    def __len__(self) -> int:
        with self._cond:
            return len(self._items)
