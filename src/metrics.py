import numpy as np

def dot_product(queries: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """
    Compute pairwise dot products between query vectors and corpus vectors.
    queries: shape (Q, D) or (D,)
    corpus: shape (N, D)
    Returns: similarity matrix of shape (Q, N) or (N,)
    """
    return np.dot(queries, corpus.T)

def cosine_similarity(queries: np.ndarray, corpus: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Compute cosine similarity with numerical stability epsilon for zero-norm vectors.
    Sim(u, v) = (u . v) / (||u|| * ||v||)
    """
    queries_2d = np.atleast_2d(queries)
    corpus_2d = np.atleast_2d(corpus)

    q_norm = np.linalg.norm(queries_2d, axis=1, keepdims=True) + eps
    c_norm = np.linalg.norm(corpus_2d, axis=1, keepdims=True) + eps

    q_normalized = queries_2d / q_norm
    c_normalized = corpus_2d / c_norm

    sims = np.dot(q_normalized, c_normalized.T)
    return sims.squeeze() if queries.ndim == 1 else sims

def euclidean_distance(queries: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """
    Compute Euclidean (L2) distance matrix using vectorized identity expansion:
    ||u - v||^2 = ||u||^2 + ||v||^2 - 2(u . v)
    """
    queries_2d = np.atleast_2d(queries)
    corpus_2d = np.atleast_2d(corpus)

    q_sq = np.sum(queries_2d ** 2, axis=1, keepdims=True)
    c_sq = np.sum(corpus_2d ** 2, axis=1, keepdims=True).T
    dists_sq = q_sq + c_sq - 2.0 * np.dot(queries_2d, corpus_2d.T)
    dists_sq = np.maximum(dists_sq, 0.0)  # Numerical non-negative clipping

    dists = np.sqrt(dists_sq)
    return dists.squeeze() if queries.ndim == 1 else dists
