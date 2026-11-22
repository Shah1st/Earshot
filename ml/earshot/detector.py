"""YAMNet (MediaPipe `yamnet.tflite`) native class scores, run with the LiteRT interpreter.

Framing contract (the app must do the same): audio arrives in 0.5 s chunks; after each chunk the
model sees the LAST 15,600 samples (zero padded at ride start). Frame i therefore ends at
t = (i + 1) * 0.5 s.
"""
from __future__ import annotations

import json

import numpy as np
from ai_edge_litert.interpreter import Interpreter

from .common import CONFIG_PATH, MODELS, SR


class Detector:
    def __init__(self, cfg: dict, model: str | None = None):
        self.model = model or cfg["audio"].get("model", "yamnet.tflite")
        model_path = str(MODELS / self.model)
        self.window = cfg["audio"]["window_samples"]
        self.hop = int(cfg["audio"]["chunk_s"] * SR)
        self.labels = (MODELS / "yamnet_labels.txt").read_text(encoding="utf-8").strip().splitlines()
        self.interp = Interpreter(model_path=model_path)
        self.inp = self.interp.get_input_details()[0]
        if list(self.inp["shape"]) != [1, self.window]:  # float YAMNet takes any length: fix it to one window
            self.interp.resize_tensor_input(self.inp["index"], [self.window])
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        outs = {int(o["shape"][-1]): o["index"] for o in self.interp.get_output_details()}
        self.out, self.emb_out = outs[521], outs.get(1024)  # embeddings only exist in the float model
        # v2: groups with "source": "head" are a logistic regression on the embedding (weights in shared/head.json)
        self.head = None
        if any(spec.get("source") == "head" for spec in cfg["groups"].values()):
            h = json.loads((CONFIG_PATH.parent / cfg["head"]["file"]).read_text(encoding="utf-8"))
            self.head = (np.asarray(h["weights"], np.float32), float(h["bias"]))
        self.groups = {g: [self.labels.index(c) for c in spec["classes"]] for g, spec in cfg["groups"].items()
                       if spec.get("source") != "head"}
        self.head_groups = [g for g, spec in cfg["groups"].items() if spec.get("source") == "head"]

    def scores(self, window: np.ndarray) -> np.ndarray:
        """One 15,600 sample float window in [-1, 1] -> 521 class scores."""
        x = np.asarray(window, dtype=np.float32).reshape(self.inp["shape"])
        self.interp.set_tensor(self.inp["index"], x)
        self.interp.invoke()
        return self.interp.get_tensor(self.out).reshape(-1).copy()

    def embedding(self) -> np.ndarray:
        """1,024 d embedding of the window just scored (float model only)."""
        return self.interp.get_tensor(self.emb_out).reshape(-1).copy()

    def score_audio(self, audio: np.ndarray, with_embeddings: bool = False):
        """Whole ride -> [n_frames, 521] (and [n_frames, 1024]), framed exactly like the streaming app."""
        n = len(audio) // self.hop
        padded = np.concatenate([np.zeros(self.window, np.float32), audio.astype(np.float32)])
        out = np.empty((n, len(self.labels)), np.float32)
        emb = np.empty((n, 1024), np.float32) if with_embeddings else None
        for i in range(n):
            end = self.window + (i + 1) * self.hop
            out[i] = self.scores(padded[end - self.window:end])
            if with_embeddings:
                emb[i] = self.embedding()
        return (out, emb) if with_embeddings else out

    def head_score(self, emb: np.ndarray, prev: np.ndarray | None = None) -> float:
        """P(distress) = sigmoid(w . [embedding now, embedding 0.5 s ago] + b); 2,048 weights, one dot product."""
        w, b = self.head
        x = np.concatenate([emb, emb if prev is None else prev])
        return float(1.0 / (1.0 + np.exp(-np.clip(x @ w + b, -30, 30))))

    def head_series(self, emb: np.ndarray) -> np.ndarray:
        """Same as head_score for a whole ride: [n, 1024] -> [n]."""
        w, b = self.head
        x = np.concatenate([emb, np.vstack([emb[:1], emb[:-1]])], axis=1)
        return 1.0 / (1.0 + np.exp(-np.clip(x @ w + b, -30, 30)))

    def group_scores(self, scores: np.ndarray, emb: np.ndarray | None = None,
                     prev: np.ndarray | None = None) -> dict[str, float]:
        out = {g: float(scores[idx].max()) for g, idx in self.groups.items()}
        for g in self.head_groups:
            out[g] = self.head_score(emb, prev)
        return out
