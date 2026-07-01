from __future__ import annotations

import logging

from .config import DEFAULT_EMBEDDING_MODEL
from .utils import cosine_similarity, hashed_vector

LOGGER = logging.getLogger(__name__)


class SemanticEmbedder:
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self._model = None
        self._available = False
        self._load_model()

    def _load_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device="cpu")
            self._available = True
            LOGGER.info("Loaded sentence-transformers model %s", self.model_name)
        except Exception as exc:  # pragma: no cover - optional dependency path
            self._model = None
            self._available = False
            LOGGER.warning("Falling back to hashed semantic vectors because embedding model could not load: %s", exc)

    @property
    def available(self) -> bool:
        return self._available

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._model is not None:
            embeddings = self._model.encode(
                texts,
                batch_size=min(64, len(texts)),
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            return [list(map(float, row)) for row in embeddings]
        return [hashed_vector(text) for text in texts]

    def similarity(self, left: str, right: str) -> float:
        vector_left, vector_right = self.embed_texts([left, right])
        return cosine_similarity(vector_left, vector_right)

    def batch_similarities(self, jd_text: str, candidate_texts: list[str]) -> list[float]:
        if not candidate_texts:
            return []
        vectors = self.embed_texts([jd_text, *candidate_texts])
        jd_vector = vectors[0]
        similarities = [cosine_similarity(jd_vector, vector) for vector in vectors[1:]]
        return [max(0.0, min(1.0, score)) for score in similarities]
