"""Transcriber port, plus a test double and the faster-whisper adapter (FR1).

The design rule this file exists to enforce: **voice is a wrapper, not a second
code path.** ASR turns audio into a string and stops. That string then goes
through exactly the same ``SyncAgent.handle_command`` that a typed command does,
so intent extraction, resolution, reconciliation and the FR5 gate are proven once
and cannot diverge between the spoken and typed routes. An agent that grew a
separate "voice pipeline" would need the safety gate proven twice, and the second
proof is the one nobody writes.

That is also why ``Transcript`` carries a confidence rather than only text: a
low-confidence transcript is a *clarification*, never a guess, and the shell can
see the difference without the agent learning about audio.

faster-whisper is not installed on this machine (NFR1's latency target cannot be
measured here). It is kept behind the same protocol as the echo double, so
enabling it is a constructor swap -- and, exactly like ``OllamaExtractor``, the
import happens at first use rather than at module import, so importing this
module can never fail because of a missing model runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable


class VoiceUnavailableError(RuntimeError):
    """A voice backend is configured but its runtime is not installed.

    Shared with ``packages/nlp/tts.py`` so a caller wiring up a voice shell has
    one exception to catch for "the machine cannot do voice", rather than one per
    vendor library.
    """


@dataclass(frozen=True)
class Transcript:
    """What the microphone said, plus how sure the recogniser is."""

    text: str
    confidence: float | None = None
    duration_sec: float | None = None
    model: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


@runtime_checkable
class Transcriber(Protocol):
    name: str

    def transcribe(self, audio: Path | str) -> Transcript: ...


@dataclass
class EchoTranscriber:
    """Test double: returns scripted text instead of decoding audio.

    Two modes, both deterministic and free. With a ``script`` it maps a path to
    the sentence that recording contains, which is how a voice test states its
    input in the same words a reader can check. Without one it reads a ``.txt``
    file, or falls back to the file stem -- so a fixture can be a text file and
    the suite needs no audio assets checked into the repository.
    """

    script: dict[str, str] = field(default_factory=dict)
    confidence: float = 1.0
    name: str = "echo"

    def transcribe(self, audio: Path | str) -> Transcript:
        key = str(audio)
        if key in self.script:
            return Transcript(self.script[key], self.confidence, model=self.name)
        path = Path(key)
        if path.suffix == ".txt" and path.exists():
            return Transcript(path.read_text(encoding="utf-8").strip(), self.confidence, model=self.name)
        return Transcript(path.stem.replace("_", " "), self.confidence, model=self.name)


class WhisperTranscriber:
    """faster-whisper adapter (STRQ: local, self-hosted, no per-call cost).

    Not exercised in CI: the library is not installed here. The constructor is
    cheap and the model is loaded on first ``transcribe`` so that building an
    agent -- including in a test that never speaks -- does not pay for, or fail
    on, a model that may not exist.
    """

    def __init__(self, model_size: str = "base.en", device: str = "cpu", beam_size: int = 5):
        self.model_size = model_size
        self.device = device
        self.beam_size = beam_size
        self.name = "faster-whisper:%s" % model_size
        self._model = None

    def transcribe(self, audio: Path | str) -> Transcript:
        model = self._load()
        segments, info = model.transcribe(str(audio), beam_size=self.beam_size)
        collected = list(segments)
        text = " ".join(segment.text.strip() for segment in collected).strip()
        # faster-whisper reports average log-probability per segment; the mean of
        # those, exponentiated, is the closest thing it offers to a confidence.
        # It is a rough signal and is labelled as one, not treated as calibrated.
        confidences = [getattr(s, "avg_logprob", None) for s in collected]
        usable = [c for c in confidences if c is not None]
        confidence = math.exp(sum(usable) / len(usable)) if usable else None
        return Transcript(
            text=text,
            confidence=confidence,
            duration_sec=getattr(info, "duration", None),
            model=self.name,
            notes=("uncalibrated confidence from avg_logprob",),
        )

    def _load(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise VoiceUnavailableError(
                    "faster-whisper is not installed, so %s cannot transcribe. "
                    "Install it with `pip install faster-whisper`, or use "
                    "EchoTranscriber for text-driven runs." % self.name
                ) from exc
            self._model = WhisperModel(self.model_size, device=self.device)
        return self._model
