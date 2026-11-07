"""High-Dimensional Vector Search Engine in Pure NumPy."""
from .index import VectorIndex
from .metrics import cosine_similarity, euclidean_distance, dot_product

__all__ = ["VectorIndex", "cosine_similarity", "euclidean_distance", "dot_product"]
