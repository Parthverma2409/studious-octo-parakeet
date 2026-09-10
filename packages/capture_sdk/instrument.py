"""Capture without touching the agent.

`SyncAgent` already emits a schema-valid trajectory on every result; the only
thing missing is somebody to hand it to. Doing that by editing `agent.py` would
put an evaluation concern inside the assistant's control flow, where it can
raise, block, or be forgotten in a new code path. So the wiring lives out here:
`instrument` returns a proxy that delegates everything and captures the
trajectory on the way out.

That also keeps Part A independent of Part B (NFR6). Nothing in
`packages/sync_agent/` imports the capture SDK, so the agent still runs, and
still passes its tests, with the evaluation platform deleted from the repo.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

#: Every entry point of `SyncAgent` that returns an `AgentResult`.
CAPTURED_METHODS = ("handle_command", "handle_email", "resolve_confirmation")


class InstrumentedAgent:
    """Delegating proxy. Anything not wrapped passes straight through."""

    def __init__(self, agent: Any, client: Any, methods: Iterable[str] = CAPTURED_METHODS):
        self._agent = agent
        self._client = client
        self._methods = frozenset(methods)

    def __getattr__(self, name: str) -> Any:
        # Guard the proxy's own state: without this, an unpickled or half-built
        # proxy looks up `_agent` through `__getattr__`, which looks up `_agent`.
        if name in ("_agent", "_client", "_methods"):
            raise AttributeError(name)
        attr = getattr(self._agent, name)
        if name in self._methods and callable(attr):
            return self._wrap(attr)
        return attr

    def __dir__(self) -> list[str]:
        return sorted(set(dir(self._agent)) | set(object.__dir__(self)))

    @property
    def agent(self) -> Any:
        """The wrapped agent, for tests that need to assert on it directly."""
        return self._agent

    def _wrap(self, method: Callable[..., Any]) -> Callable[..., Any]:
        def instrumented(*args: Any, **kwargs: Any) -> Any:
            result = method(*args, **kwargs)
            self._capture(result)
            return result

        return instrumented

    def _capture(self, result: Any) -> None:
        """Best effort, on purpose.

        An exception raised here would surface as a failure of a booking the
        agent actually completed, so a broken capture client degrades the
        evaluation record and nothing else. The agent's own exceptions are not
        caught: they propagate exactly as they would uninstrumented, and no
        half-built trajectory is captured for them.
        """
        trajectory = getattr(result, "trajectory", None)
        if not isinstance(trajectory, dict) or not trajectory:
            return
        try:
            self._client.capture(trajectory)
        except Exception:  # noqa: BLE001 -- see docstring
            pass


def instrument(agent: Any, client: Any, methods: Iterable[str] = CAPTURED_METHODS) -> InstrumentedAgent:
    return InstrumentedAgent(agent, client, methods)
