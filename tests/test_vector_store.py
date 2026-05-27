"""
tests/test_vector_store.py
Tests for the fraud case vector store and similarity search.
"""

import pytest
import numpy as np
import os
import tempfile
from pathlib import Path

os.environ.setdefault("GROQ_API_KEY", "test_key")


# ── Fixtures ───────────────────────────────────────────────────────────────────

def make_record(
    case_id="case_1",
    fraud_score=0.8,
    amount=1000.0,
    category="atm_cash",
    location="unknown",
    action="BLOCK",
    merchant_risk=0.7,
    geo_score=0.2,
    device_match=False,
    velocity=5,
    behaviour=0.9,
    truth="FRAUD",
):
    from pipeline.vector_store import CaseRecord
    return CaseRecord(
        case_id            = case_id,
        transaction_id     = case_id,
        fraud_score        = fraud_score,
        amount             = amount,
        merchant_category  = category,
        location           = location,
        recommended_action = action,
        merchant_risk      = merchant_risk,
        geo_score          = geo_score,
        device_match       = device_match,
        velocity_hour      = velocity,
        behaviour_score    = behaviour,
        ground_truth       = truth,
    )


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Fresh vector store using a temp directory."""
    from pipeline import vector_store as vs_module
    monkeypatch.setattr(vs_module, "VECTOR_STORE_PATH", tmp_path / "vs")
    monkeypatch.setattr(vs_module, "INDEX_PATH",        tmp_path / "vs" / "faiss_index.bin")
    monkeypatch.setattr(vs_module, "METADATA_PATH",     tmp_path / "vs" / "case_metadata.json")
    from pipeline.vector_store import FraudVectorStore
    return FraudVectorStore()


# ── Unit tests ─────────────────────────────────────────────────────────────────

class TestCaseRecord:
    def test_to_dict_and_back(self):
        from pipeline.vector_store import CaseRecord
        r = make_record()
        d = r.to_dict()
        r2 = CaseRecord.from_dict(d)
        assert r2.case_id     == r.case_id
        assert r2.fraud_score == r.fraud_score

    def test_timestamp_set_automatically(self):
        r = make_record()
        assert r.timestamp is not None
        assert "T" in r.timestamp  # ISO format


class TestVectorStore:
    def test_starts_empty(self, store):
        assert store.size() == 0

    def test_add_case_increments_size(self, store):
        store.add_case(make_record("c1"))
        store.add_case(make_record("c2"))
        assert store.size() == 2

    def test_search_returns_empty_when_no_cases(self, store):
        results = store.search(make_record("query"), k=3)
        assert results == []

    def test_search_finds_similar_case(self, store):
        # Add a clearly fraudulent case
        fraud_record = make_record(
            "fraud_1", fraud_score=0.9, amount=5000.0,
            category="atm_cash", location="unknown",
            action="BLOCK", truth="FRAUD"
        )
        store.add_case(fraud_record)

        # Query with similar characteristics
        query = make_record(
            "query", fraud_score=0.85, amount=4500.0,
            category="atm_cash", location="unknown",
            action="REVIEW", truth="UNKNOWN"
        )
        results = store.search(query, k=3, min_score=0.0)
        assert len(results) >= 1
        assert results[0]["transaction_id"] == "fraud_1"

    def test_search_excludes_self(self, store):
        r = make_record("self_case")
        store.add_case(r)
        results = store.search(r, k=3, min_score=0.0)
        ids = [res["transaction_id"] for res in results]
        assert "self_case" not in ids

    def test_search_respects_k(self, store):
        for i in range(10):
            store.add_case(make_record(f"case_{i}", fraud_score=0.8))
        query   = make_record("query", fraud_score=0.8)
        results = store.search(query, k=3, min_score=0.0)
        assert len(results) <= 3

    def test_similar_fraud_ranks_higher_than_legit(self, store):
        # Add a fraud case and a legit case with same category
        fraud_rec = make_record(
            "fraud", fraud_score=0.9, amount=5000.0,
            category="atm_cash", device_match=False,
            behaviour=0.9, truth="FRAUD", action="BLOCK"
        )
        legit_rec = make_record(
            "legit", fraud_score=0.1, amount=50.0,
            category="retail", device_match=True,
            behaviour=0.1, truth="LEGITIMATE", action="ALLOW"
        )
        store.add_case(fraud_rec)
        store.add_case(legit_rec)

        query = make_record(
            "q", fraud_score=0.88, amount=4800.0,
            category="atm_cash", device_match=False,
            behaviour=0.85, truth="UNKNOWN"
        )
        results = store.search(query, k=2, min_score=0.0)
        assert len(results) >= 1
        assert results[0]["transaction_id"] == "fraud"

    def test_embedding_correct_dimension(self, store):
        from pipeline.vector_store import EMBEDDING_DIM
        r   = make_record()
        emb = store._record_to_embedding(r)
        assert emb.shape == (EMBEDDING_DIM,)
        assert emb.dtype == np.float32

    def test_fraud_score_in_embedding(self, store):
        r1 = make_record(fraud_score=0.9)
        r2 = make_record(fraud_score=0.1)
        e1 = store._record_to_embedding(r1)
        e2 = store._record_to_embedding(r2)
        assert e1[0] == pytest.approx(0.9, abs=0.01)
        assert e2[0] == pytest.approx(0.1, abs=0.01)

    def test_save_and_reload(self, store, tmp_path, monkeypatch):
        from pipeline import vector_store as vs_module
        monkeypatch.setattr(vs_module, "VECTOR_STORE_PATH", tmp_path / "vs")
        monkeypatch.setattr(vs_module, "METADATA_PATH",     tmp_path / "vs" / "case_metadata.json")
        monkeypatch.setattr(vs_module, "INDEX_PATH",        tmp_path / "vs" / "faiss_index.bin")

        store.add_case(make_record("save_test_1"))
        store.add_case(make_record("save_test_2"))
        store.save()

        from pipeline.vector_store import FraudVectorStore
        store2 = FraudVectorStore()
        assert store2.size() == 2

    def test_similarity_score_in_results(self, store):
        store.add_case(make_record("c1"))
        results = store.search(make_record("q"), k=1, min_score=0.0)
        if results:
            assert "similarity" in results[0]
            assert 0.0 <= results[0]["similarity"] <= 1.0

    def test_min_score_filter(self, store):
        store.add_case(make_record("c1", fraud_score=0.5))
        # Use very high min_score — should filter everything out
        results = store.search(make_record("q", fraud_score=0.9), k=3, min_score=0.9999)
        assert len(results) == 0
