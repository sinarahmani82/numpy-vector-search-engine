import pytest
import numpy as np
from src.metrics import cosine_similarity, euclidean_distance, dot_product
from src.index import VectorIndex

def test_exact_identity_retrieval():
    """Verify that querying with an exact indexed vector returns itself with similarity 1.0."""
    dim = 64
    np.random.seed(42)
    corpus = np.random.randn(100, dim).astype(np.float32)
    
    index = VectorIndex(dimension=dim, metric="cosine")
    index.add(corpus, ids=[f"doc_{i}" for i in range(100)])

    # Query with exactly the 42nd vector
    query = corpus[42]
    top_ids, top_scores = index.search(query, top_k=1)

    assert top_ids[0] == "doc_42"
    assert pytest.approx(top_scores[0], rel=1e-5) == 1.0

def test_euclidean_distance_non_negativity():
    """Verify that Euclidean distance satisfies non-negativity and self-distance is zero."""
    dim = 32
    v1 = np.random.randn(1, dim).astype(np.float32)
    v2 = np.random.randn(5, dim).astype(np.float32)

    self_dist = euclidean_distance(v1, v1)
    assert pytest.approx(self_dist, abs=1e-5) == 0.0

    pair_dists = euclidean_distance(v1, v2)
    assert np.all(pair_dists >= 0.0)

def test_top_k_monotonic_ordering():
    """Ensure that returned top-K results are strictly sorted monotonically."""
    dim = 16
    index = VectorIndex(dimension=dim, metric="cosine")
    index.add(np.random.randn(50, dim))

    query = np.random.randn(dim)
    _, scores = index.search(query, top_k=10)

    # For cosine similarity, scores must be non-increasing
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1]
