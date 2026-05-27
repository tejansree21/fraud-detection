"""
pipeline/vector_store.py
Vector store for similar fraud case retrieval.

Uses FAISS (Facebook AI Similarity Search) for fast nearest-neighbour
search over transaction embeddings. No external API needed — runs fully local.

How it works:
  1. Each flagged transaction is converted to a feature vector
  2. Vectors are indexed in FAISS
  3. At report time, query the index for the K most similar past cases
  4. Results are passed to the Case Reporter for context

Embedding strategy:
  - Uses the same preprocessed features as the ML models (15 dims)
  - Augmented with fraud score and enrichment signals (5 dims)
  - Total: 20-dim embedding, fast and effective for similarity search
"""

from __future__ import annotations
import json
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any

logger = logging.getLogger("vector_store")

VECTOR_STORE_PATH = Path("model_artifacts/vector_store")
INDEX_PATH        = VECTOR_STORE_PATH / "faiss_index.bin"
METADATA_PATH     = VECTOR_STORE_PATH / "case_metadata.json"

EMBEDDING_DIM = 20   # feature vector dimension


@dataclass
class CaseRecord:
    """A stored fraud case with its embedding and metadata."""
    case_id:        str
    transaction_id: str
    fraud_score:    float
    amount:         float
    merchant_category: str
    location:       str
    recommended_action: str
    merchant_risk:  float
    geo_score:      float
    device_match:   bool
    velocity_hour:  int
    behaviour_score: float
    ground_truth:   str = "UNKNOWN"   # FRAUD / LEGITIMATE / UNKNOWN
    timestamp:      str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CaseRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class FraudVectorStore:
    """
    FAISS-backed vector store for similar fraud case retrieval.

    Falls back gracefully to a simple numpy-based search if FAISS
    is not installed (slower but functional).
    """

    def __init__(self):
        self.embeddings: list[np.ndarray] = []
        self.records:    list[CaseRecord] = []
        self._faiss_index = None
        self._use_faiss   = self._try_import_faiss()
        VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
        self._load()

    def _try_import_faiss(self) -> bool:
        try:
            import faiss  # noqa
            logger.info("FAISS available — using fast vector search")
            return True
        except ImportError:
            logger.info("FAISS not installed — using numpy fallback search")
            return False

    # ── Public interface ───────────────────────────────────────────────────────

    def add_case(
        self,
        record:    CaseRecord,
        embedding: np.ndarray | None = None,
    ) -> None:
        """Add a fraud case to the vector store."""
        if embedding is None:
            embedding = self._record_to_embedding(record)

        embedding = embedding.astype(np.float32)
        if embedding.shape[0] != EMBEDDING_DIM:
            embedding = self._pad_or_truncate(embedding, EMBEDDING_DIM)

        self.embeddings.append(embedding)
        self.records.append(record)

        # Rebuild FAISS index periodically
        if self._use_faiss and len(self.embeddings) % 50 == 0:
            self._build_faiss_index()

    def search(
        self,
        query_record: CaseRecord,
        k:            int = 3,
        min_score:    float = 0.5,
    ) -> list[dict[str, Any]]:
        """
        Find k most similar past fraud cases.
        Returns list of dicts with case metadata and similarity score.
        """
        if len(self.embeddings) == 0:
            return []

        query_emb = self._record_to_embedding(query_record).astype(np.float32)

        if self._use_faiss and self._faiss_index is not None:
            results = self._search_faiss(query_emb, k)
        else:
            results = self._search_numpy(query_emb, k)

        # Filter by minimum similarity and exclude the query itself
        filtered = [
            r for r in results
            if r["similarity"] >= min_score
            and r["transaction_id"] != query_record.transaction_id
        ]

        return filtered[:k]

    def size(self) -> int:
        return len(self.records)

    def save(self) -> None:
        """Persist index and metadata to disk."""
        if not self.records:
            return

        # Save metadata
        metadata = [r.to_dict() for r in self.records]
        METADATA_PATH.write_text(json.dumps(metadata, indent=2))

        # Save embeddings
        if self.embeddings:
            np.save(str(VECTOR_STORE_PATH / "embeddings.npy"),
                    np.stack(self.embeddings))

        if self._use_faiss and self._faiss_index is not None:
            import faiss
            faiss.write_index(self._faiss_index, str(INDEX_PATH))

        logger.debug("Vector store saved: %d cases", len(self.records))

    # ── Embedding construction ─────────────────────────────────────────────────

    def _record_to_embedding(self, record: CaseRecord) -> np.ndarray:
        """
        Convert a CaseRecord to a 20-dim float32 embedding.

        Features:
          [0]  fraud_score
          [1]  amount_log (log-normalised)
          [2]  merchant_risk_score
          [3]  geo_consistency_score
          [4]  device_match (0/1)
          [5]  velocity_hour (normalised)
          [6]  user_behaviour_score
          [7-11] merchant_category one-hot
          [12-16] location one-hot
          [17] action_block (0/1)
          [18] action_review (0/1)
          [19] is_fraud (0/1 if known)
        """
        vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)

        vec[0] = float(record.fraud_score)
        vec[1] = float(np.log1p(record.amount) / 12.0)  # normalise log amount
        vec[2] = float(record.merchant_risk)
        vec[3] = float(record.geo_score)
        vec[4] = float(record.device_match)
        vec[5] = min(float(record.velocity_hour) / 10.0, 1.0)
        vec[6] = float(record.behaviour_score)

        # Merchant category one-hot (dims 7-11)
        cat_map = {"retail": 7, "atm_cash": 8, "bank_transfer": 9,
                   "online": 10, "unknown": 11}
        cat_idx = cat_map.get(record.merchant_category, 11)
        vec[cat_idx] = 1.0

        # Location one-hot (dims 12-16)
        loc_map = {"Dublin, IE": 12, "London, UK": 13, "New York, US": 14,
                   "unknown": 15}
        loc_idx = loc_map.get(record.location, 16)
        if loc_idx < EMBEDDING_DIM:
            vec[loc_idx] = 1.0

        # Action (dims 17-18)
        vec[17] = 1.0 if record.recommended_action == "BLOCK"  else 0.0
        vec[18] = 1.0 if record.recommended_action == "REVIEW" else 0.0

        # Ground truth (dim 19)
        vec[19] = 1.0 if record.ground_truth == "FRAUD" else 0.0

        return vec

    # ── Search backends ────────────────────────────────────────────────────────

    def _search_faiss(self, query: np.ndarray, k: int) -> list[dict]:
        import faiss
        k_actual = min(k, len(self.embeddings))
        query_2d = query.reshape(1, -1)

        # Normalise for cosine similarity
        faiss.normalize_L2(query_2d)
        distances, indices = self._faiss_index.search(query_2d, k_actual)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.records):
                continue
            record = self.records[idx]
            results.append({
                **record.to_dict(),
                "similarity": round(float(dist), 4),
            })
        return results

    def _search_numpy(self, query: np.ndarray, k: int) -> list[dict]:
        """Cosine similarity search using numpy."""
        if not self.embeddings:
            return []

        matrix = np.stack(self.embeddings)

        # Normalise
        query_norm  = query / (np.linalg.norm(query) + 1e-8)
        matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)

        similarities = matrix_norm @ query_norm
        k_actual     = min(k, len(similarities))
        top_indices  = np.argsort(similarities)[::-1][:k_actual]

        results = []
        for idx in top_indices:
            record = self.records[idx]
            results.append({
                **record.to_dict(),
                "similarity": round(float(similarities[idx]), 4),
            })
        return results

    def _build_faiss_index(self) -> None:
        try:
            import faiss
            matrix = np.stack(self.embeddings).astype(np.float32)
            faiss.normalize_L2(matrix)
            index = faiss.IndexFlatIP(EMBEDDING_DIM)  # inner product = cosine after normalisation
            index.add(matrix)
            self._faiss_index = index
            logger.debug("FAISS index rebuilt: %d vectors", index.ntotal)
        except Exception as e:
            logger.warning("FAISS index build failed: %s", e)

    def _pad_or_truncate(self, vec: np.ndarray, dim: int) -> np.ndarray:
        if len(vec) >= dim:
            return vec[:dim]
        return np.pad(vec, (0, dim - len(vec)))

    def _load(self) -> None:
        """Load existing index from disk if available."""
        if not METADATA_PATH.exists():
            return
        try:
            metadata  = json.loads(METADATA_PATH.read_text())
            self.records = [CaseRecord.from_dict(m) for m in metadata]

            emb_path = VECTOR_STORE_PATH / "embeddings.npy"
            if emb_path.exists():
                matrix = np.load(str(emb_path))
                self.embeddings = [matrix[i] for i in range(len(matrix))]

            if self._use_faiss and INDEX_PATH.exists() and self.embeddings:
                import faiss
                self._faiss_index = faiss.read_index(str(INDEX_PATH))

            logger.info("Vector store loaded: %d cases", len(self.records))
        except Exception as e:
            logger.warning("Could not load vector store: %s — starting fresh", e)
            self.records    = []
            self.embeddings = []
