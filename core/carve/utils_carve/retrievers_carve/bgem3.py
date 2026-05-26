"""
BGEM3-based retriever for use inside the cco package.

This is a simplified dense+hybrid retriever built on top of BGEM3FlagModel.
It provides a minimal API for the retrieval_replay_fewshot baseline.
"""

from typing import List
import os

import numpy as np
from FlagEmbedding import BGEM3FlagModel
from sklearn.metrics.pairwise import cosine_similarity


class BGEM3Retriever:
    """
    Minimal BGEM3 retriever:
    - fit(corpus): compute and cache corpus embeddings
    - search_indices(query, top_k): return top-k indices by cosine similarity
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", use_fp16: bool = True, device: str = "cuda"):
        cache_dir = os.environ.get("HF_HOME", None)
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16, devices=str(device), cache_dir=cache_dir)
        self.corpus: List[str] = []
        self.corpus_embeddings = None

    def fit(self, corpus: List[str], batch_size: int = 32):
        """
        Compute dense embeddings for the corpus.
        """
        self.corpus = corpus
        embeddings = self.model.encode(
            corpus,
            batch_size=batch_size,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        self.corpus_embeddings = embeddings["dense_vecs"]

    def search_indices(self, query: str, top_k: int = 1) -> np.ndarray:
        """
        Return indices of top-k most similar documents using cosine similarity.
        """
        if self.corpus_embeddings is None:
            raise ValueError("Must call fit() before search_indices().")

        query_embedding = self.model.encode(
            [query],
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )["dense_vecs"]

        similarities = cosine_similarity(query_embedding, self.corpus_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return top_indices

