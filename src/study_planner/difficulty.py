"""Transformer-backed subject difficulty classification."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Protocol


DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

DEFAULT_DIFFICULTY_REFERENCES: dict[str, list[str]] = {
    "hard": [
        "neural networks",
        "transformers",
        "backpropagation",
        "attention mechanism",
        "deep learning",
        "natural language processing",
        "gradient descent",
        "embeddings",
    ],
    "medium": [
        "databases",
        "SQL queries",
        "network security",
        "encryption",
        "protocols",
        "data modeling",
    ],
    "easy": [
        "basic programming",
        "introduction to computing",
        "fundamentals",
        "overview",
        "history of computers",
    ],
}


class EmbeddingModel(Protocol):
    def encode(self, texts: Iterable[str]):
        """Return vector embeddings for the supplied texts."""


@dataclass(frozen=True)
class DifficultyResult:
    subject: str
    predicted: str
    scores: dict[str, float]

    def to_tool_message(self) -> str:
        ordered_scores = ", ".join(
            f"{level}: {score:.3f}" for level, score in self.scores.items()
        )
        return f"{self.subject} -> {self.predicted} ({ordered_scores})"


class SubjectDifficultyClassifier:
    """Classify subjects through cosine similarity against reference topics."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        references: Mapping[str, list[str]] | None = None,
        model: EmbeddingModel | None = None,
    ) -> None:
        self.model_name = model_name
        self.references = dict(references or DEFAULT_DIFFICULTY_REFERENCES)
        self._model = model
        self._reference_embeddings: dict[str, list[list[float]]] | None = None

    @property
    def model(self) -> EmbeddingModel:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def classify(self, subject_name: str) -> DifficultyResult:
        subject = normalize_subject_name(subject_name)
        subject_embedding = _first_vector(self.model.encode([subject]))
        scores: dict[str, float] = {}
        for level, reference_vectors in self.reference_embeddings.items():
            scores[level] = max(_cosine_similarity(subject_embedding, vector) for vector in reference_vectors)
        predicted = max(scores, key=scores.get)
        return DifficultyResult(subject=subject, predicted=predicted, scores=scores)

    @property
    def reference_embeddings(self) -> dict[str, list[list[float]]]:
        if self._reference_embeddings is None:
            self._reference_embeddings = {
                level: _vectors(self.model.encode(examples))
                for level, examples in self.references.items()
            }
        return self._reference_embeddings


def normalize_subject_name(subject_name: str) -> str:
    return " ".join(subject_name.strip().split())


def _vectors(raw_vectors) -> list[list[float]]:
    if hasattr(raw_vectors, "tolist"):
        raw_vectors = raw_vectors.tolist()
    return [list(map(float, vector)) for vector in raw_vectors]


def _first_vector(raw_vectors) -> list[float]:
    return _vectors(raw_vectors)[0]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)
