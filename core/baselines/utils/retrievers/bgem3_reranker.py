import numpy as np
from FlagEmbedding import FlagModel, BGEM3FlagModel, FlagReranker
from sklearn.metrics.pairwise import cosine_similarity
import os
from functools import lru_cache
from scipy.sparse import csr_matrix
import torch
from tqdm import tqdm

class CorpusRetriever:
    def __init__(self, model_name='BAAI/bge-base-en-v1.5'):
        """
        Initialize the retriever with a FlagEmbedding model.
        Popular models:
        - 'BAAI/bge-base-en-v1.5' (recommended for general use)
        - 'BAAI/bge-large-en-v1.5' (better performance, slower)
        - 'BAAI/bge-small-en-v1.5' (faster, smaller)
        """
        self.model = FlagModel(model_name,
                              query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                              use_fp16=True,
                              cache_dir=os.environ["HF_HOME"])  # Use fp16 for faster inference
        self.corpus = None
        self.corpus_embeddings = None

    def fit(self, corpus):
        """
        Fit the retriever on a corpus by computing embeddings.

        Args:
            corpus (list): List of strings to search through
        """
        self.corpus = corpus
        
        # Compute embeddings for the corpus
        self.corpus_embeddings = self.model.encode(corpus)
        
    def search(self, query, top_k=1):
        """
        Search for the most similar documents to a query.

        Args:
            query (str): Query string
            top_k (int): Number of top results to return

        Returns:
            list: List of tuples (document, similarity_score, index)
        """
        if self.corpus_embeddings is None:
            raise ValueError("Must call fit() first to compute corpus embeddings")

        # Encode the query
        query_embedding = self.model.encode([query])

        # Compute similarities
        similarities = cosine_similarity(query_embedding, self.corpus_embeddings)[0]

        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                'document': self.corpus[idx],
                'similarity': similarities[idx],
                'index': idx
            })

        return [result['document'] for result in results[:top_k]]

class CorpusRetrieverWithReranker:
    def __init__(self,
                 retrieval_model='BAAI/bge-m3',
                 reranker_model='BAAI/bge-reranker-v2-m3',
                 use_fp16=True):
        """
        Initialize retriever with BGE-M3 and reranker.

        Args:
            retrieval_model (str): BGE-M3 model for initial retrieval
            reranker_model (str): Reranker model for final ranking
            use_fp16 (bool): Use fp16 for faster inference
        """
        self.retrieval_model = BGEM3FlagModel(retrieval_model, use_fp16=use_fp16)

        self.reranker = FlagReranker(reranker_model, use_fp16=use_fp16)

        self.corpus = None
        self.corpus_embeddings = None

    def fit(self, corpus, batch_size=32):
        """
        Fit the retriever on a corpus by computing embeddings.

        Args:
            corpus (list): List of strings to search through
            batch_size (int): Batch size for embedding computation
        """
        self.corpus = corpus
        
        # BGE-M3 supports multiple retrieval methods
        # We'll use dense embeddings for initial retrieval
        embeddings = self.retrieval_model.encode(
            corpus,
            batch_size=batch_size,
            return_dense=True,  # Dense embeddings for similarity search
            return_sparse=False,  # Can enable for hybrid search
            return_colbert_vecs=False  # Can enable for fine-grained matching
        )

        self.corpus_embeddings = embeddings['dense_vecs']
        
    def search(self, query, top_k=1, initial_candidates=50):
        """
        Search using two-stage retrieval: initial retrieval + reranking.

        Args:
            query (str): Query string
            top_k (int): Number of final results to return
            initial_candidates (int): Number of candidates for initial retrieval

        Returns:
            list: List of dictionaries with reranked results
        """
        if self.corpus_embeddings is None:
            raise ValueError("Must call fit() first to compute corpus embeddings")

        # Stage 1: Initial retrieval with BGE-M3
        query_embedding = self.retrieval_model.encode(
            [query],
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )['dense_vecs']

        # Get initial candidates
        similarities = cosine_similarity(query_embedding, self.corpus_embeddings)[0]
        initial_indices = np.argsort(similarities)[::-1][:initial_candidates]

        # Prepare candidates for reranking
        candidates = [self.corpus[idx] for idx in initial_indices]

        # Stage 2: Reranking
        rerank_scores = self.reranker.compute_score(
            [[query, candidate] for candidate in candidates],
            batch_size=32
        )

        # Combine results
        results = []
        for i, idx in enumerate(initial_indices):
            results.append({
                'document': self.corpus[idx],
                'retrieval_similarity': similarities[idx],
                'rerank_score': rerank_scores[i],
                'index': idx
            })

        # Sort by rerank scores and return top-k
        results.sort(key=lambda x: x['rerank_score'], reverse=True)

        return [result['document'] for result in results[:top_k]]

    def search_dense_only(self, query, top_k=1):
        """
        Search using only dense embeddings (faster, no reranking).
        """
        if self.corpus_embeddings is None:
            raise ValueError("Must call fit() first to compute corpus embeddings")

        query_embedding = self.retrieval_model.encode(
            [query],
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )['dense_vecs']

        similarities = cosine_similarity(query_embedding, self.corpus_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                'document': self.corpus[idx],
                'similarity': similarities[idx],
                'index': idx
            })

        return results


class HybridBGEM3Retriever:
    """
    Optimized retriever using BGE-M3's hybrid capabilities (dense + sparse).
    Optimized for pandas apply() usage with query caching and vectorized operations.
    """
    def __init__(self, corpus, model_name='BAAI/bge-m3', use_fp16=True):
        print("Building Hybrid BGEM3 index...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16, devices=str(self.device), cache_dir=os.environ["HF_HOME"])
        self.reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=use_fp16, devices=str(self.device), cache_dir=os.environ["HF_HOME"])
        self.corpus = corpus
        
        # Optimization: Cache for query embeddings
        self._query_cache = {}
        self._cache_size = 1000  # Limit cache size
        
        # Pre-computed sparse similarity structures
        self._sparse_matrix = None
        self._vocab_to_idx = None
        
        # Now fit the model
        self.fit()

    def fit(self, batch_size=128):
        """Compute both dense and sparse embeddings for hybrid search."""
        self.embeddings = self.model.encode(
            self.corpus,
            batch_size=batch_size,
            return_dense=True,
            return_sparse=True,  # Enable sparse embeddings
            return_colbert_vecs=False
        )
        print("Index built with {} documents.".format(len(self.corpus)))
        # Optimization: Pre-build sparse matrix for faster computation
        self._build_sparse_matrix()
        print("Sparse matrix built for hybrid search.")
        
    def _build_sparse_matrix(self):
        """Build a sparse matrix representation for faster sparse similarity computation."""
        # Collect all unique tokens
        all_tokens = set()
        for doc_sparse in self.embeddings['lexical_weights']:
            all_tokens.update(doc_sparse.keys())
        
        # Create vocabulary mapping
        self._vocab_to_idx = {token: idx for idx, token in enumerate(sorted(all_tokens))}
        vocab_size = len(self._vocab_to_idx)
        
        # Build sparse matrix (documents x vocabulary)
        rows, cols, data = [], [], []
        for doc_idx, doc_sparse in enumerate(self.embeddings['lexical_weights']):
            for token, weight in doc_sparse.items():
                rows.append(doc_idx)
                cols.append(self._vocab_to_idx[token])
                data.append(float(weight))  # Convert to Python float to avoid float16 issues
        
        self._sparse_matrix = csr_matrix((data, (rows, cols)), 
                                       shape=(len(self.corpus), vocab_size))
    
    def _get_query_embeddings(self, query):
        """Get query embeddings with caching."""
        if query in self._query_cache:
            return self._query_cache[query]
        
        # Clear cache if it gets too large
        if len(self._query_cache) >= self._cache_size:
            # Remove oldest entries (simple FIFO)
            keys_to_remove = list(self._query_cache.keys())[:self._cache_size // 2]
            for key in keys_to_remove:
                del self._query_cache[key]
        
        query_embeddings = self.model.encode(
            [query],
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        
        self._query_cache[query] = query_embeddings
        return query_embeddings
    
    def _compute_sparse_similarity_vectorized(self, query_sparse):
        """Vectorized sparse similarity computation."""
        # Convert query sparse to vector
        query_vector = np.zeros(len(self._vocab_to_idx))
        for token, weight in query_sparse.items():
            if token in self._vocab_to_idx:
                query_vector[self._vocab_to_idx[token]] = weight
        
        # Compute similarities using matrix multiplication
        sparse_similarities = self._sparse_matrix.dot(query_vector)
        return sparse_similarities
        
    def retrieve_model_card(self, instruction, top_k=1, alpha=0.7, initial_candidates=50):
        """
        Optimized hybrid search combining dense and sparse similarities with reranking.

        Args:
            alpha (float): Weight for dense similarity (1-alpha for sparse)
        """
        # Get cached query embeddings
        query_embeddings = self._get_query_embeddings(instruction)

        # Dense similarity
        dense_similarities = cosine_similarity(
            query_embeddings['dense_vecs'],
            self.embeddings['dense_vecs']
        )[0]

        # Optimized sparse similarity computation
        query_sparse = query_embeddings['lexical_weights'][0]
        sparse_similarities = self._compute_sparse_similarity_vectorized(query_sparse)

        # Normalize similarities
        dense_similarities = (dense_similarities - dense_similarities.min()) / \
                           (dense_similarities.max() - dense_similarities.min() + 1e-8)

        if sparse_similarities.max() > sparse_similarities.min():
            sparse_similarities = (sparse_similarities - sparse_similarities.min()) / \
                                (sparse_similarities.max() - sparse_similarities.min() + 1e-8)

        # Combine similarities
        hybrid_similarities = alpha * dense_similarities + (1 - alpha) * sparse_similarities

        # Get initial candidates
        initial_indices = np.argsort(hybrid_similarities)[::-1][:initial_candidates]
        candidates = [self.corpus[idx] for idx in initial_indices]

        # Rerank
        rerank_scores = self.reranker.compute_score(
            [[instruction, candidate] for candidate in candidates]
        )

        # Prepare results
        results = []
        for i, idx in enumerate(initial_indices):
            results.append({
                'document': self.corpus[idx],
                'hybrid_similarity': hybrid_similarities[idx],
                'dense_similarity': dense_similarities[idx],
                'sparse_similarity': sparse_similarities[idx],
                'rerank_score': rerank_scores[i],
                'index': idx
            })

        results.sort(key=lambda x: x['rerank_score'], reverse=True)
        if top_k > 1:
            return [result['document'] for result in results[:top_k]]
        else:
            return results[0]['document'] if results else None

    def retrieve_model_card_with_oracle(self, instruction, domain, top_k=1):
        """
        Oracle retrieval with the same pipeline as retrieve_model_card,
        but restricted to the domain-filtered corpus.
        """
        filtered_corpus = [doc for doc in self.corpus if f'"domain": "{domain}"' in doc]
        print(f"Oracle retrieval: {len(filtered_corpus)} documents match domain '{domain}' out of {len(self.corpus)} total.")
        if not filtered_corpus:
            return self.retrieve_model_card(instruction, top_k=top_k)

        query_embeddings = self._get_query_embeddings(instruction)
        filtered_embeddings = self.model.encode(
            filtered_corpus,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        dense_similarities = cosine_similarity(
            query_embeddings['dense_vecs'],
            filtered_embeddings['dense_vecs']
        )[0]

        query_sparse = query_embeddings['lexical_weights'][0]
        all_tokens = set()
        for doc_sparse in filtered_embeddings['lexical_weights']:
            all_tokens.update(doc_sparse.keys())

        vocab_to_idx = {token: idx for idx, token in enumerate(sorted(all_tokens))}
        rows, cols, data = [], [], []
        for doc_idx, doc_sparse in enumerate(filtered_embeddings['lexical_weights']):
            for token, weight in doc_sparse.items():
                rows.append(doc_idx)
                cols.append(vocab_to_idx[token])
                data.append(float(weight))

        sparse_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(len(filtered_corpus), len(vocab_to_idx))
        )

        query_vector = np.zeros(len(vocab_to_idx))
        for token, weight in query_sparse.items():
            if token in vocab_to_idx:
                query_vector[vocab_to_idx[token]] = weight

        sparse_similarities = sparse_matrix.dot(query_vector)

        dense_similarities = (dense_similarities - dense_similarities.min()) / \
            (dense_similarities.max() - dense_similarities.min() + 1e-8)

        if sparse_similarities.max() > sparse_similarities.min():
            sparse_similarities = (sparse_similarities - sparse_similarities.min()) / \
                (sparse_similarities.max() - sparse_similarities.min() + 1e-8)

        hybrid_similarities = 0.7 * dense_similarities + 0.3 * sparse_similarities

        k_candidates = min(50, len(filtered_corpus))
        initial_indices = np.argsort(hybrid_similarities)[::-1][:k_candidates]
        candidates = [filtered_corpus[idx] for idx in initial_indices]

        rerank_scores = self.reranker.compute_score(
            [[instruction, candidate] for candidate in candidates]
        )

        results = []
        for i, idx in enumerate(initial_indices):
            results.append({
                'document': filtered_corpus[idx],
                'hybrid_similarity': hybrid_similarities[idx],
                'dense_similarity': dense_similarities[idx],
                'sparse_similarity': sparse_similarities[idx],
                'rerank_score': rerank_scores[i],
                'index': idx
            })

        results.sort(key=lambda x: x['rerank_score'], reverse=True)

        if top_k > 1:
            return [result['document'] for result in results[:top_k]]
        return results[0]['document'] if results else None

    
    def clear_cache(self):
        """Clear the query cache to free memory."""
        self._query_cache.clear()
    
    def search_batch(self, queries, top_k=1, alpha=0.7, initial_candidates=50):
        """
        Batch search for multiple queries - more efficient for large batches.
        
        Args:
            queries (list): List of query strings
            top_k (int): Number of results per query
            alpha (float): Weight for dense similarity
            initial_candidates (int): Number of candidates for reranking
            
        Returns:
            list: List of results for each query
        """
        # Encode all queries at once
        batch_query_embeddings = self.model.encode(
            queries,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False
        )
        
        results = []
        for i, query in tqdm(enumerate(queries), total=len(queries), desc="FLAGEMBEDDING retrieval"):
            # Extract embeddings for this query
            query_dense = batch_query_embeddings['dense_vecs'][i:i+1]
            query_sparse = batch_query_embeddings['lexical_weights'][i]
            
            # Dense similarity
            dense_similarities = cosine_similarity(
                query_dense,
                self.embeddings['dense_vecs']
            )[0]
            
            # Sparse similarity
            sparse_similarities = self._compute_sparse_similarity_vectorized(query_sparse)
            
            # Normalize and combine
            dense_similarities = (dense_similarities - dense_similarities.min()) / \
                               (dense_similarities.max() - dense_similarities.min() + 1e-8)
            
            if sparse_similarities.max() > sparse_similarities.min():
                sparse_similarities = (sparse_similarities - sparse_similarities.min()) / \
                                    (sparse_similarities.max() - sparse_similarities.min() + 1e-8)
            
            hybrid_similarities = alpha * dense_similarities + (1 - alpha) * sparse_similarities
            
            # Get candidates and rerank
            initial_indices = np.argsort(hybrid_similarities)[::-1][:initial_candidates]
            candidates = [self.corpus[idx] for idx in initial_indices]
            
            rerank_scores = self.reranker.compute_score(
                [[query, candidate] for candidate in candidates]
            )
            
            # Sort by rerank scores
            query_results = []
            for j, idx in enumerate(initial_indices):
                query_results.append({
                    'document': self.corpus[idx],
                    'hybrid_similarity': hybrid_similarities[idx],
                    'dense_similarity': dense_similarities[idx],
                    'sparse_similarity': sparse_similarities[idx],
                    'rerank_score': rerank_scores[j],
                    'index': idx
                })
            
            query_results.sort(key=lambda x: x['rerank_score'], reverse=True)
            if top_k > 1:
                results.append([result['document'] for result in query_results[:top_k]])
            else:
                results.append(query_results[0]['document'] if query_results else None)
        
        return results

