"""Voice shell (FR1, NFR1): the null adapters, and graceful degradation.

The property this file exists to protect is architectural: **voice is a wrapper,
not a second code path.** The shell turns audio into a string and hands that
string to the same `handle_command` a typed command uses, so the FR5 gate,
reconciliation and intent extraction are proven once. A spoken command that could
reach the calendar by some other route would need every safety property proven
twice, and the second proof is the one nobody writes.

The second property is availability: neither faster-whisper nor Piper is
installed on this machine. A missing model runtime must produce a clear,
actionable error at the moment someone tries to speak -- never an ImportError at
module import, which would take the whole text path down with it.
"""

from __future__ import annotations

import builtins
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

from packages.core.models import BookingType, CalendarEvent
from packages.nlp.asr import EchoTranscriber, Transcriber, Transcript, VoiceUnavailableError, WhisperTranscriber
from packages.nlp.tts import (
    DEFAULT_MIN_CONFIDENCE,
    NullSpeaker,
    PiperSpeaker,
    Speaker,
    VoiceShell,
    VoiceTurn,
)
from packages.providers.fake_calendar import FakeCalendar
from packages.sync_agent.agent import SyncAgent

NOW = datetime(2026, 8, 18, 8, 0)


def _calendar_with_flight() -> FakeCalendar:
    calendar = FakeCalendar()
    calendar.seed(
        CalendarEvent(
            event_id="E1",
            # Titled the way the user refers to it: "cancel my Chennai flight"
            # is matched against the calendar entry's own words (FR3).
            title="my Chennai flight",
            start_time=datetime(2026, 8, 20, 9, 15),
            booking_ref="AI440-XY12",
            booking_type=BookingType.FLIGHT,
        ).with_alarm_offset(60, "AL-E1")
    )
    return calendar


def _shell(text: str, confidence: float = 1.0, calendar: FakeCalendar | None = None):
    calendar = calendar if calendar is not None else _calendar_with_flight()
    agent = SyncAgent(calendar)
    transcriber = EchoTranscriber(script={"clip.wav": text}, confidence=confidence)
    speaker = NullSpeaker()
    return calendar, agent, speaker, VoiceShell(transcriber, speaker, agent)


# -- the null adapters ------------------------------------------------------


def test_the_null_adapters_satisfy_the_protocols() -> None:
    """A test double that does not implement the port proves nothing."""
    assert isinstance(EchoTranscriber(), Transcriber)
    assert isinstance(NullSpeaker(), Speaker)
    assert isinstance(WhisperTranscriber(), Transcriber)
    assert isinstance(PiperSpeaker("voice.onnx"), Speaker)


def test_the_echo_transcriber_returns_its_script() -> None:
    transcript = EchoTranscriber(script={"clip.wav": "cancel my Chennai flight"}).transcribe("clip.wav")
    assert transcript.text == "cancel my Chennai flight"
    assert transcript.is_empty is False


def test_the_echo_transcriber_reads_a_text_fixture(tmp_path: Path) -> None:
    """So a voice fixture can be a .txt file and the repo needs no audio assets."""
    fixture = tmp_path / "utterance.txt"
    fixture.write_text("book a flight to Chennai\n", encoding="utf-8")
    assert EchoTranscriber().transcribe(fixture).text == "book a flight to Chennai"


def test_the_null_speaker_records_what_would_have_been_said() -> None:
    """Which is what makes the shell's refusal paths observable at all."""
    speaker = NullSpeaker()
    speaker.speak("Added your flight.")
    speaker.speak("Anything else?")
    assert speaker.spoken == ["Added your flight.", "Anything else?"]


def test_a_transcript_of_whitespace_counts_as_empty() -> None:
    assert Transcript("   \n").is_empty is True
    assert Transcript("x").is_empty is False


# -- the shell round-trips through the text path ---------------------------


def test_a_spoken_command_round_trips_through_the_ordinary_text_path() -> None:
    """FR1. Audio in, the same agent, the reply spoken back."""
    calendar, _agent, speaker, shell = _shell(
        "book a flight to Chennai on 21 August at 9am", calendar=FakeCalendar()
    )

    turn = shell.handle("clip.wav", NOW)

    assert turn.reached_agent is True
    assert turn.result.status == "success"
    assert [m[0] for m in calendar.mutations] == ["CREATE"]
    assert speaker.spoken == [turn.result.message]
    assert "Chennai" in speaker.spoken[0]


def test_a_spoken_query_is_answered_without_touching_the_calendar() -> None:
    """FR6 through the microphone. A question is not an instruction."""
    calendar, _agent, speaker, shell = _shell("when is my Chennai flight")

    turn = shell.handle("clip.wav", NOW)

    assert turn.result.status == "success"
    assert "Chennai" in speaker.spoken[0]
    assert calendar.mutations == []


def test_a_spoken_destructive_command_still_stops_at_the_confirmation_gate() -> None:
    """The gate must not be reachable around, only through.

    This is the whole reason the shell owns no booking logic: FR5 is enforced in
    exactly one place, and speaking is not a way to get past it.
    """
    calendar, agent, speaker, shell = _shell("cancel my Chennai flight")

    turn = shell.handle("clip.wav", NOW)

    assert turn.result.status == "awaiting_confirmation"
    assert calendar.mutations == []
    assert "E1" in calendar.events
    # The prompt is what was spoken -- the user is asked out loud, as UC3 says.
    assert speaker.spoken == [turn.result.message]
    assert "Delete" in speaker.spoken[0]

    confirmed = agent.resolve_confirmation(turn.result.confirmation.request_id, True, NOW)
    assert confirmed.status == "success"
    assert calendar.mutations == [("DELETE", "E1")]


def test_the_shell_holds_no_booking_logic_of_its_own() -> None:
    """Structural check on the design rule.

    A VoiceShell that reached the calendar, the reconciler or the confirmation
    store directly would be a second code path with its own safety story.
    """
    import ast
    import inspect

    from packages.nlp import tts

    source = inspect.getsource(tts)
    names = {n.attr for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Attribute)}
    for forbidden in ("calendar", "confirmations", "engine", "resolve_confirmation", "apply_destructive"):
        assert forbidden not in names, "the voice shell reaches into %s" % forbidden


# -- the shell refuses rather than guesses ---------------------------------


def test_an_empty_transcript_never_reaches_the_agent() -> None:
    """Silence is not a command."""
    calendar, _agent, speaker, shell = _shell("")

    turn = shell.handle("clip.wav", NOW)

    assert turn.reached_agent is False
    assert turn.result is None
    assert calendar.mutations == []
    assert "did not catch" in speaker.spoken[0]


def test_a_low_confidence_transcript_is_repeated_back_not_acted_on() -> None:
    """NFR1's real risk is not latency, it is a half-heard sentence.

    'Cancel my flight' is one dropped syllable from something the user did not
    say, and the NLU layer will happily produce a confident frame from noise. So
    a low-confidence transcript is quoted back for correction and the agent never
    sees it.
    """
    calendar, _agent, speaker, shell = _shell("cancel my Chennai flight", confidence=0.2)

    turn = shell.handle("clip.wav", NOW)

    assert turn.reached_agent is False
    assert calendar.mutations == []
    assert "not sure" in speaker.spoken[0]
    assert "cancel my Chennai flight" in speaker.spoken[0]


def test_the_confidence_threshold_is_a_boundary_not_a_range() -> None:
    """Exactly at the threshold the command is accepted; just below it is not."""
    _c1, _a1, _s1, at_threshold = _shell("when is my Chennai flight", confidence=DEFAULT_MIN_CONFIDENCE)
    _c2, _a2, _s2, below = _shell("when is my Chennai flight", confidence=DEFAULT_MIN_CONFIDENCE - 0.01)

    assert at_threshold.handle("clip.wav", NOW).reached_agent is True
    assert below.handle("clip.wav", NOW).reached_agent is False


def test_a_transcriber_that_reports_no_confidence_is_trusted() -> None:
    """Absent is not the same as low. A backend that cannot score its output
    must not be treated as though it scored 0 and be silenced entirely."""
    agent = SyncAgent(_calendar_with_flight())
    speaker = NullSpeaker()

    class Unscored:
        name = "unscored"

        def transcribe(self, audio):
            return Transcript("when is my Chennai flight", confidence=None)

    turn = VoiceShell(Unscored(), speaker, agent).handle("clip.wav", NOW)
    assert turn.reached_agent is True


def test_a_voice_turn_records_what_was_heard_and_what_was_said() -> None:
    """FR7: a spoken interaction has to be reconstructable after the fact."""
    _calendar, _agent, _speaker, shell = _shell("when is my Chennai flight")
    turn = shell.handle("clip.wav", NOW)

    assert isinstance(turn, VoiceTurn)
    assert turn.transcript.text == "when is my Chennai flight"
    assert turn.spoken == turn.result.message


# -- degrading gracefully ---------------------------------------------------


def test_importing_the_voice_modules_never_requires_the_optional_libraries() -> None:
    """The text path must not go down because a model runtime is absent.

    An import-time `import faster_whisper` would make `packages.nlp.asr`
    unimportable here -- and `packages.sync_agent.agent` imports the NLP package,
    so a top-level import would take the whole agent down with it.

    Run in a subprocess with a clean interpreter, deliberately: reloading the
    modules in-process rebinds their classes and silently breaks every `isinstance`
    and `pytest.raises` in this file, which is a worse test than no test.
    """
    assert "faster_whisper" not in sys.modules

    script = (
        "import sys;"
        "import packages.nlp.asr, packages.nlp.tts, packages.sync_agent.agent;"
        "print('faster_whisper' in sys.modules, 'piper' in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "False False"


def test_constructing_the_real_transcriber_is_free_and_does_not_load_a_model() -> None:
    """So building an agent that never speaks costs nothing and cannot fail."""
    transcriber = WhisperTranscriber("base.en")
    assert transcriber.name == "faster-whisper:base.en"


def test_the_real_transcriber_raises_an_actionable_error_when_the_library_is_missing() -> None:
    """Not ImportError, and not a stack trace ending in someone else's module.

    The message has to name the library, the thing that failed, and the way out.
    """
    with pytest.raises(VoiceUnavailableError) as raised:
        WhisperTranscriber("base.en").transcribe("clip.wav")

    message = str(raised.value)
    assert "faster-whisper" in message
    assert "pip install faster-whisper" in message
    assert "EchoTranscriber" in message


def test_the_real_transcriber_error_survives_a_library_that_imports_but_breaks() -> None:
    """A half-installed library raises ImportError from inside its own import.

    Catching only "module not found" would let that escape as something the
    caller cannot act on.
    """
    real_import = builtins.__import__

    def exploding(name, *args, **kwargs):
        if name == "faster_whisper":
            raise ImportError("libcudnn.so.9: cannot open shared object file")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = exploding
    try:
        with pytest.raises(VoiceUnavailableError, match="not installed"):
            WhisperTranscriber().transcribe("clip.wav")
    finally:
        builtins.__import__ = real_import


def test_the_real_speaker_raises_an_actionable_error_when_the_binary_is_missing() -> None:
    """Piper is a binary on PATH, not an import; the check has to match that."""
    with pytest.raises(VoiceUnavailableError) as raised:
        PiperSpeaker("voice.onnx", binary="piper-not-installed-anywhere").speak("hello")

    message = str(raised.value)
    assert "piper-not-installed-anywhere" in message
    assert "PATH" in message
    assert "NullSpeaker" in message


def test_the_real_speaker_checks_its_voice_model_too(tmp_path: Path, monkeypatch) -> None:
    """A present binary with an absent voice fails just as clearly."""
    monkeypatch.setattr("packages.nlp.tts.shutil.which", lambda _: str(tmp_path / "piper"))

    with pytest.raises(VoiceUnavailableError, match="not found"):
        PiperSpeaker(str(tmp_path / "missing.onnx")).speak("hello")


def test_a_missing_voice_backend_does_not_stop_the_typed_command_path() -> None:
    """The point of degrading gracefully: the assistant still works, silently.

    An unavailable microphone or speaker must cost the user voice, not the
    product.
    """
    calendar = _calendar_with_flight()
    agent = SyncAgent(calendar)
    shell = VoiceShell(WhisperTranscriber(), PiperSpeaker("voice.onnx"), agent)

    with pytest.raises(VoiceUnavailableError):
        shell.handle("clip.wav", NOW)

    # The same agent, driven by text, is entirely unaffected.
    assert agent.handle_command("when is my Chennai flight", NOW).status == "success"
