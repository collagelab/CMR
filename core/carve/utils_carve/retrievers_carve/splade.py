"""
SPLADE retriever implementation for use inside the cco package.

This is adapted from the original implementation in data/utils/retrievers/splade.py,
but lives under cco/ so it is importable when using the installed cco package.
"""

from typing import List
import os

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM


class SpladeRetriever:
    """
    SPLADE-based retriever.

    Provides:
    - encode_splade_batch: encode a corpus into sparse vectors
    - encode_splade: encode a single query into a sparse vector
    - search_indices: return top-k indices for a query given precomputed doc vectors
    """

    def __init__(
        self,
        query_model_id: str = "naver/efficient-splade-VI-BT-large-query",
        doc_model_id: str = "naver/efficient-splade-VI-BT-large-doc",
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        cache_dir = os.environ.get("HF_HOME", None)

        self.query_tokenizer = AutoTokenizer.from_pretrained(query_model_id, cache_dir=cache_dir)
        self.doc_tokenizer = AutoTokenizer.from_pretrained(doc_model_id, cache_dir=cache_dir)
        self.query_model = AutoModelForMaskedLM.from_pretrained(query_model_id, cache_dir=cache_dir).to(
            self.device
        ).eval()
        self.doc_model = AutoModelForMaskedLM.from_pretrained(doc_model_id, cache_dir=cache_dir).to(
            self.device
        ).eval()

        self._doc_vecs = None

    def max_pool(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Perform SPLADEv2 max pooling operation.
        Returns a vocabulary-sized tensor with max-pooled values.
        """
        return torch.log1p(torch.relu(logits)).amax(dim=0)

    def encode_splade(self, text: str) -> torch.Tensor:
        """Encodes text into a sparse vector representation using SPLADE."""
        tokens = self.query_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length",
        ).to(self.device)
        with torch.no_grad():
            logits = self.query_model(**tokens).logits[0]  # tensor of shape [N, vocab]

        return self.max_pool(logits)  # tensor of size vocab

    def encode_splade_batch(self, texts: List[str], batch_size: int = 64) -> torch.Tensor:
        """Encodes a batch of texts into sparse vectors using SPLADE."""
        all_vecs = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            tokens = self.doc_tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding="max_length",
            ).to(self.device)

            with torch.no_grad():
                logits = self.doc_model(**tokens).logits

            # Apply SPLADE max pooling for each sequence in the batch
            batch_vecs = []
            for j in range(logits.size(0)):
                batch_vecs.append(self.max_pool(logits[j]))
            all_vecs.append(torch.stack(batch_vecs))

        doc_vecs = torch.cat(all_vecs, dim=0)  # shape [num_texts, vocab_size]
        self._doc_vecs = doc_vecs
        return doc_vecs

    def fit(self, corpus: List[str], batch_size: int = 64):
        """Encode and cache document vectors for the given corpus."""
        self.encode_splade_batch(corpus, batch_size=batch_size)

    def search_indices(self, query: str, top_k: int = 1) -> torch.Tensor:
        """
        Return indices of top-k most similar documents using SPLADE dot-product similarity.
        Assumes fit() has been called and _doc_vecs is populated.
        """
        if self._doc_vecs is None:
            raise ValueError("Must call fit() before search_indices().")

        query_vec = self.encode_splade(query).unsqueeze(0)
        scores = torch.mv(self._doc_vecs, query_vec.squeeze(0))
        top_indices = torch.argsort(scores, descending=True)[:top_k]
        return top_indices

