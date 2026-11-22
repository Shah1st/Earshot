"""Continuous on-device speech recognition with Vosk (0 LLM tokens).

The recogniser runs on every 0.5 s chunk and gives the pipeline two things:
  * a rolling transcript kept in memory as text only: at most the last 120 s (asr.gps_context_s), older words
    are dropped; the judge gets the last 45 s for sound events and the last 120 s for GPS events;
  * keyword triggers: a safety phrase appearing in the transcript opens an event even when nobody shouts.

Keywords are phrases matched in the FREE transcript. The grammar limited pass from the first plan was
dropped: measured on the tune dialogues in driving noise it invents keywords in 48% of harmless
conversations, while phrase matching on the free transcript hits 100% of threat dialogues and 15% of
normal ones (all real mentions such as "the police closed the road", which the LLM then dismisses).
"""
from __future__ import annotations

import json

import numpy as np

from .common import MODELS, SR, to_pcm16


def find_keywords(text: str, keywords: list[str]) -> dict[str, int]:
    """Phrase counts in a transcript, longest phrases first so "help me" is not also counted as "help"."""
    text, out = f" {text} ", {}
    for k in sorted(keywords, key=len, reverse=True):
        n = text.count(f" {k} ")
        if n:
            out[k] = n
            text = text.replace(f" {k} ", " # ")
    return out


class StreamAsr:
    """Live recogniser. feed() every chunk; transcript(t0, t1) any time."""

    _models: dict = {}

    def __init__(self, cfg: dict, keep_all: bool = False):
        """keep_all: never forget words. Only for offline recording of a whole ride (RecordedAsr); the phone keeps 120 s."""
        from vosk import KaldiRecognizer, Model, SetLogLevel

        self.keep_all = keep_all

        SetLogLevel(-1)
        self.cfg = cfg["asr"]
        name = self.cfg["model"]
        if name not in StreamAsr._models:
            StreamAsr._models[name] = Model(str(MODELS / name))
        self.rec = KaldiRecognizer(StreamAsr._models[name], SR)
        self.rec.SetWords(True)
        self.words: list[tuple[str, float, float]] = []  # (word, start_s, end_s), confident words of finished utterances
        self.partial = ""
        self.fired: dict[str, int] = {}

    def _new_hits(self, text: str) -> list[str]:
        hits = []
        for k, n in find_keywords(text, self.cfg["keywords"]).items():
            if n > self.fired.get(k, 0):
                hits.append(k)
                self.fired[k] = n
        return hits

    def _take(self, result: dict) -> list[str]:
        words = [w for w in result.get("result", []) if w["conf"] >= self.cfg["min_word_conf"]]
        self.words += [(w["word"], w["start"], w["end"]) for w in words]
        if self.words and not self.keep_all:  # rolling memory: forget everything older than the longest context the judge can be given
            keep = self.words[-1][2] - max(self.cfg["context_s"], self.cfg["gps_context_s"])
            self.words = [w for w in self.words if w[2] > keep]
        hits = self._new_hits(" ".join(w["word"] for w in words))
        self.partial, self.fired = "", {}
        return hits

    def feed(self, now: float, chunk: np.ndarray) -> list[str]:
        """Returns the keywords newly heard in this chunk."""
        if self.rec.AcceptWaveform(to_pcm16(chunk).tobytes()):  # an utterance just ended
            return self._take(json.loads(self.rec.Result()))
        self.partial = json.loads(self.rec.PartialResult()).get("partial", "")
        return self._new_hits(self.partial)

    def finish(self) -> list[str]:
        return self._take(json.loads(self.rec.FinalResult()))

    def transcript(self, t0: float, t1: float) -> str:
        done = " ".join(w for w, a, b in self.words if b > t0 and a <= t1)
        return f"{done} {self.partial}".strip()


class RecordedAsr:
    """Replay of a StreamAsr run over a whole ride (cached, so evaluation does not redo recognition)."""

    def __init__(self, record: dict, chunk_s: float = 0.5):
        self.hits = {int(k): v for k, v in record["hits"].items()}
        self.words = [tuple(w) for w in record["words"]]
        self.chunk_s = chunk_s

    @staticmethod
    def record(cfg: dict, audio: np.ndarray) -> dict:
        asr, hop, hits = StreamAsr(cfg, keep_all=True), int(cfg["audio"]["chunk_s"] * SR), {}
        for i in range(len(audio) // hop):
            h = asr.feed((i + 1) * cfg["audio"]["chunk_s"], audio[i * hop:(i + 1) * hop])
            if h:
                hits[i] = h
        asr.finish()
        return {"hits": hits, "words": asr.words}

    def feed(self, now: float, chunk: np.ndarray) -> list[str]:
        return self.hits.get(round(now / self.chunk_s) - 1, [])

    def transcript(self, t0: float, t1: float) -> str:
        return " ".join(w for w, a, b in self.words if b > t0 and a <= t1)
