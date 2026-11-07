import numpy as np
from typing import List, Tuple, Any, Optional
from src.metrics import cosine_similarity, euclidean_distance, dot_product

class VectorIndex:
    """High-dimensional dense vector index supporting sub-linear top-K partition search."""
    def __init__(self, dimension: int, metric: str = "cosine"):
        if dimension <= 0:
            raise ValueError("Vector dimensionality must be a positive integer.")
        if metric not in ["cosine", "euclidean", "dot"]:
            raise ValueError(f"Unsupported metric '{metric}'. Use 'cosine', 'euclidean', or 'dot'.")

        self.dimension = dimension
        self.metric = metric
        self.vectors: np.ndarray = np.empty((0, dimension), dtype=np.float32)
        self.ids: List[Any] = []

    def add(self, vectors: np.ndarray, ids: Optional[List[Any]] = None) -> None:
        """Append a batch of vectors into the search index."""
        vectors_arr = np.asarray(vectors, dtype=np.float32)
        if vectors_arr.ndim != 2 or vectors_arr.shape[1] != self.dimension:
            raise ValueError(f"Input vectors must have shape (N, {self.dimension}).")

        n_new = vectors_arr.shape[0]
        if ids is None:
            current_count = len(self.ids)
            assigned_ids = list(range(current_count, current_count + n_new))
        else:
            if len(ids) != n_new:
                raise ValueError("Number of IDs must strictly match number of vectors.")
            assigned_ids = ids

        self.vectors = np.vstack([self.vectors, vectors_arr])
        self.ids.extend(assigned_ids)

    def search(self, query: np.ndarray, top_k: int = 5) -> Tuple[List[Any], np.ndarray]:
        """
        Search for top-K closest vectors using partial partition ranking for O(N) performance.
        Returns: (ranked_ids, ranked_scores)
        """
        if len(self.ids) == 0:
            return [], np.array([], dtype=np.float32)

        k = min(top_k, len(self.ids))
        q = np.asarray(query, dtype=np.float32)
        if q.shape[-1] != self.dimension:
            raise ValueError(f"Query vector dimension must match {self.dimension}.")

        # 1. Compute similarity/distance scores
        if self.metric == "cosine":
            scores = cosine_similarity(q, self.vectors)
            descending = True
        elif self.metric == "dot":
            scores = dot_product(q, self.vectors)
            descending = True
        else:  # Euclidean
            scores = euclidean_distance(q, self.vectors)
            descending = False

        # 2. Sub-linear top-K partitioning via quickselect
        if descending:
            partition_idx = np.argpartition(-scores, k - 1)[:k]
            sorted_order = partition_idx[np.argsort(-scores[partition_idx])]
        else:
            partition_idx = np.argpartition(scores, k - 1)[:k]
            sorted_order = partition_idx[np.argsort(scores[partition_idx])]

        top_ids = [self.ids[idx] for idx in sorted_order]
        top_scores = scores[sorted_order]
        return top_ids, top_scores

    def __len__(self) -> int:
        return len(self.ids)
