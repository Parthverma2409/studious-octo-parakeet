"""Speaker port, a recording test double, the Piper adapter, and the voice shell.

``VoiceShell`` is the whole Phase 5 voice story in one small class, and it is
deliberately small: transcribe, hand the resulting **text** to the ordinary
command handler, speak whatever the handler said. It owns no booking logic, no
intent rules and no calendar access, so there is no behaviour that a spoken
command can reach and a typed one cannot -- including no way around the FR5
confirmation gate.

Two refusals are implemented here rather than in the agent, because they are
properties of the microphone and not of the domain:

  * an empty transcript is not a command, and
  * a transcript below ``min_confidence`` is a request for clarification.

Both stop before the handler is called. Feeding the agent a half-heard sentence
would let the NLU layer make a confident booking out of noise, and "cancel my
flight" is one dropped syllable away from something the user did not say.

Piper is not installed here, so ``PiperSpeaker`` degrades the same way
``WhisperTranscriber`` does: the check happens at first use, never at import.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

from .asr import Transcriber, Transcript, VoiceUnavailableError

#: Below this, the shell asks again instead of acting. Chosen to be visible and
#: adjustable rather than tuned -- calibrating it needs the ASR-noise slice that
#: this machine cannot run, and a fabricated threshold would look like evidence.
DEFAULT_MIN_CONFIDENCE = 0.5


@runtime_checkable
class Speaker(Protocol):
    name: str

    def speak(self, text: str) -> None: ...


@runtime_checkable
class CommandHandler(Protocol):
    """Structural view of ``SyncAgent.handle_command``.

    Structural on purpose: ``packages/nlp`` must not import ``packages/sync_agent``
    (the agent imports the NLP layer, so the reverse would be a cycle), and the
    shell genuinely does not care what produced the reply.
    """

    def handle_command(self, text: str, now: datetime): ...


@dataclass
class NullSpeaker:
    """Records what would have been said. The default outside a real device.

    Tests assert on ``spoken``, which is also what makes the shell's refusal
    paths observable without an audio device or a listener.
    """

    spoken: list[str] = field(default_factory=list)
    name: str = "null"

    def speak(self, text: str) -> None:
        self.spoken.append(text)


class PiperSpeaker:
    """Piper TTS adapter (local, self-hosted, no per-utterance cost).

    Not exercised in CI: the binary is not on this machine. Synthesis writes a
    WAV rather than playing it, because playback is the one part of this that
    genuinely cannot be tested headlessly, and mixing "make the audio" with "make
    a noise" would drag that untestability into the adapter.
    """

    def __init__(self, model_path: str, binary: str = "piper", out_dir: Path | str = "."):
        self.model_path = model_path
        self.binary = binary
        self.out_dir = Path(out_dir)
        self.name = "piper:%s" % Path(model_path).stem
        self._seq = 0

    def speak(self, text: str) -> None:
        executable = shutil.which(self.binary)
        if executable is None:
            raise VoiceUnavailableError(
                "the '%s' binary is not on PATH, so %s cannot speak. Install Piper "
                "from https://github.com/rhasspy/piper, or use NullSpeaker for "
                "text-only runs." % (self.binary, self.name)
            )
        if not Path(self.model_path).exists():
            raise VoiceUnavailableError(
                "Piper voice model %r not found; download a .onnx voice or use NullSpeaker."
                % self.model_path
            )
        self._seq += 1
        self.out_dir.mkdir(parents=True, exist_ok=True)
        output = self.out_dir / ("utterance-%03d.wav" % self._seq)
        subprocess.run(
            [executable, "--model", self.model_path, "--output_file", str(output)],
            input=text.encode("utf-8"),
            check=True,
        )


@dataclass(frozen=True)
class VoiceTurn:
    """One spoken interaction, for the trajectory and for tests.

    ``result`` is None exactly when the shell refused before reaching the
    handler, which is the distinction that matters when reading a transcript of a
    session: nothing was asked of the agent, so nothing could have been changed.
    """

    transcript: Transcript
    spoken: str
    result: object = None

    @property
    def reached_agent(self) -> bool:
        return self.result is not None


@dataclass
class VoiceShell:
    """Audio in -> the ordinary text command path -> audio out (FR1)."""

    transcriber: Transcriber
    speaker: Speaker
    handler: CommandHandler
    min_confidence: float = DEFAULT_MIN_CONFIDENCE

    def handle(self, audio: Path | str, now: datetime) -> VoiceTurn:
        transcript = self.transcriber.transcribe(audio)

        if transcript.is_empty:
            return self._refuse(transcript, "I did not catch anything. Could you say that again?")

        if transcript.confidence is not None and transcript.confidence < self.min_confidence:
            # Quote it back. A user correcting a misheard sentence is a far
            # better outcome than the agent acting on its best guess.
            return self._refuse(
                transcript,
                "I think you said '%s', but I am not sure. Could you repeat that?" % transcript.text,
            )

        result = self.handler.handle_command(transcript.text, now)
        reply = getattr(result, "message", "")
        self.speaker.speak(reply)
        return VoiceTurn(transcript=transcript, spoken=reply, result=result)

    def _refuse(self, transcript: Transcript, reply: str) -> VoiceTurn:
        self.speaker.speak(reply)
        return VoiceTurn(transcript=transcript, spoken=reply, result=None)
