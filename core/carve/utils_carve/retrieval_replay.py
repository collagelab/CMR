"""
Retrieval Replay Few-Shot Baseline Components

This module provides:
- PromptReplayBuffer: Maintains cumulative replay buffer for continual learning
- RetrieverWrapper: Unified interface for all retrievers with metadata support
- ExperienceIndex: Manages experience-specific indices with metadata
"""

import random
import hashlib
from typing import List, Dict, Optional, Any, Tuple
import torch
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
from .retrievers_carve.splade import SpladeRetriever
from .retrievers_carve.bgem3 import BGEM3Retriever
import os
import re


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison by:
    - Converting to lowercase
    - Stripping leading/trailing whitespace
    - Normalizing whitespace (collapsing multiple spaces/newlines to single space)
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text string
    """
    if not isinstance(text, str):
        text = str(text)
    # Convert to lowercase
    text = text.lower()
    # Strip leading/trailing whitespace
    text = text.strip()
    # Normalize whitespace: collapse multiple spaces/newlines/tabs to single space
    text = re.sub(r'\s+', ' ', text)
    return text


class PromptReplayBuffer:
    """
    Maintains a cumulative replay buffer for continual learning.
    Samples a percentage of examples from each experience and stores them cumulatively.
    """
    
    def __init__(self, replay_ratio: float = 0.1, seed: Optional[int] = None):
        """
        Initialize the replay buffer.
        
        Args:
            replay_ratio: Percentage of examples to sample from each experience (default 0.1 = 10%)
            seed: Random seed for reproducibility
        """
        self.replay_ratio = replay_ratio
        self.seed = seed
        self.examples: List[Dict[str, Any]] = []
        
    def add_experience(self, experience_examples: List[Dict[str, Any]], experience_name: str):
        """
        Sample replay_ratio portion from this experience and append to buffer.
        
        Args:
            experience_examples: List of examples from the experience
            experience_name: Name of the experience (e.g., "apibench", "mllm")
        """
        if not experience_examples:
            return
        
        # Calculate number of samples
        num_samples = max(1, int(len(experience_examples) * self.replay_ratio))
        num_samples = min(num_samples, len(experience_examples))
        
        # Sample with seed for reproducibility
        if self.seed is not None:
            random.seed(self.seed)
        
        sampled_indices = random.sample(range(len(experience_examples)), num_samples)
        sampled_examples = [experience_examples[i] for i in sampled_indices]
        
        # Add experience_name to each example if not present
        for ex in sampled_examples:
            if 'experience_id' not in ex:
                ex['experience_id'] = experience_name
        
        self.examples.extend(sampled_examples)
        
        print(f"  Added {num_samples} examples from {experience_name} to replay buffer "
              f"(total buffer size: {len(self.examples)})")
    
    def get_examples(self) -> List[Dict[str, Any]]:
        """Get all examples in the replay buffer."""
        return self.examples
    
    def clear(self):
        """Clear the replay buffer."""
        self.examples = []


def generate_example_id(instruction: str, model_name: str, experience_name: str, local_idx: int) -> str:
    """
    Generate a stable unique example_id for a training example.
    
    Args:
        instruction: The prompt/instruction text
        model_name: The target model name
        experience_name: Name of the experience
        local_idx: Local index within the experience
        
    Returns:
        Unique example_id string
    """
    # Use a combination that ensures uniqueness
    return f"{experience_name}_{local_idx}"


class RetrieverWrapper:
    """
    Unified interface wrapping existing retrievers with metadata support and self-masking.
    """
    
    def __init__(self, retriever_type: str, device: Optional[str] = None):
        """
        Initialize the retriever wrapper.
        
        Args:
            retriever_type: One of "bm25", "sentence_transformer", "splade", "flagembedding"
            device: Device to use (for GPU-based retrievers)
        """
        self.retriever_type = retriever_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.corpus: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self._retriever = None
        self._fitted = False
        
    def fit(self, corpus: List[str], metadata_list: List[Dict[str, Any]]):
        """
        Build index over corpus with associated metadata.
        
        Args:
            corpus: List of prompt strings to index
            metadata_list: List of metadata dicts (one per corpus item)
                Each dict should contain: example_id, model_id, experience_id, model_card_snippet
        """
        if len(corpus) != len(metadata_list):
            raise ValueError(f"Corpus length ({len(corpus)}) must match metadata length ({len(metadata_list)})")
        
        self.corpus = corpus
        self.metadata = metadata_list
        
        if self.retriever_type == "bm25":
            tokenized_corpus = [doc.split(" ") for doc in corpus]
            self._retriever = BM25Okapi(tokenized_corpus)
            
        elif self.retriever_type == "sentence_transformer":
            model = SentenceTransformer(
                "all-mpnet-base-v2", 
                device=self.device, 
                cache_folder=os.environ.get("HF_HOME", None)
            )
            list_of_text = [text.replace("\n", " ") for text in corpus]
            
            # Check for prompt length issues - all-mpnet-base-v2 has max_seq_length=512
            # Count tokens to detect truncation (approximate: ~4 chars per token)
            max_seq_length = model.get_max_seq_length()
            long_prompts = []
            for i, text in enumerate(list_of_text):
                approx_tokens = len(text) // 4  # Rough estimate
                if approx_tokens > max_seq_length:
                    long_prompts.append((i, approx_tokens, text[:100]))
            
            if long_prompts:
                print(f"  [WARNING] {len(long_prompts)} prompts exceed max_seq_length={max_seq_length} tokens")
                print(f"    These will be truncated during encoding, which may affect retrieval quality")
                print(f"    Sample long prompt (idx={long_prompts[0][0]}, ~{long_prompts[0][1]} tokens): {long_prompts[0][2]}...")
                if len(long_prompts) > 1:
                    print(f"    ... and {len(long_prompts)-1} more")
            
            self._corpus_embeddings = model.encode(list_of_text, convert_to_tensor=True)
            self._model = model
            self._retriever = model
            self._max_seq_length = max_seq_length
            
        elif self.retriever_type == "splade":
            # SPLADE retriever using internal cco implementation
            splade = SpladeRetriever(device=self.device)
            splade.fit(corpus, batch_size=64)
            self._retriever = splade
        elif self.retriever_type == "flagembedding":
            # BGEM3-based retriever using internal cco implementation
            bge = BGEM3Retriever(device=self.device)
            bge.fit(corpus, batch_size=32)
            self._retriever = bge
        else:
            raise ValueError(f"Unknown retriever type: {self.retriever_type}")
        
        self._fitted = True
    
    def retrieve_with_metadata(
        self, 
        query: str, 
        top_k: int, 
        exclude_example_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k similar prompts with metadata, optionally excluding certain example_ids.
        Also excludes candidates whose normalized prompt text matches the normalized query text
        (to avoid trivial retrieval when same prompt appears with different IDs).
        
        Args:
            query: Query prompt string
            top_k: Number of results to return
            exclude_example_ids: List of example_ids to exclude (for self-masking)
            
        Returns:
            List of dicts with keys: 'prompt', 'model_id', 'example_id', 'experience_id', 'model_card_snippet'
        """
        if not self._fitted:
            raise ValueError("Must call fit() before retrieve_with_metadata()")
        
        exclude_example_ids = exclude_example_ids or []
        exclude_set = set(exclude_example_ids)
        
        # Normalize query text for text-level exclusion
        normalized_query = normalize_text(query)
        
        # Retrieve more than top_k to account for filtering (both ID and text-level)
        retrieve_k = top_k * 3 if (exclude_example_ids or normalized_query) else top_k
        
        if self.retriever_type == "bm25":
            tokenized_query = query.split(" ")
            scores = self._retriever.get_scores(tokenized_query)
            top_indices = np.argsort(scores)[::-1][:retrieve_k]
            
        elif self.retriever_type == "sentence_transformer":
            # Check if query exceeds max_seq_length (will be truncated)
            max_seq_length = getattr(self._model, 'get_max_seq_length', lambda: 512)()
            query_approx_tokens = len(query) // 4  # Rough estimate
            if query_approx_tokens > max_seq_length:
                import warnings
                warnings.warn(
                    f"Query prompt exceeds max_seq_length={max_seq_length} tokens (~{query_approx_tokens} tokens). "
                    f"Will be truncated during encoding, which may affect retrieval quality. "
                    f"Query preview: {query[:100]}..."
                )
            
            query_embedding = self._model.encode(query, convert_to_tensor=True, device=self.device)
            similarities = util.cos_sim(query_embedding, self._corpus_embeddings)[0]
            top_indices = torch.argsort(similarities, descending=True)[:retrieve_k].cpu().numpy()
            
        elif self.retriever_type == "splade":
            # Use internal SPLADE retriever
            top_indices = self._retriever.search_indices(query, top_k=retrieve_k).cpu().numpy()
            
        elif self.retriever_type == "flagembedding":
            # Use internal BGEM3 retriever
            top_indices = self._retriever.search_indices(query, top_k=retrieve_k)
            
        else:
            raise ValueError(f"Unknown retriever type: {self.retriever_type}")
        
        # Filter out excluded example_ids and text-level duplicates, then build results
        results = []
        for idx in top_indices:
            metadata = self.metadata[idx]
            example_id = metadata.get('example_id')
            
            # Skip if excluded by ID
            if example_id in exclude_set:
                continue
            
            # Skip if excluded by normalized text (text-level exclusion)
            candidate_prompt = self.corpus[idx]
            normalized_candidate = normalize_text(candidate_prompt)
            if normalized_candidate == normalized_query:
                continue
            
            results.append({
                'prompt': candidate_prompt,
                'model_id': metadata.get('model_id', ''),
                'example_id': example_id,
                'experience_id': metadata.get('experience_id', ''),
                'model_card_snippet': metadata.get('model_card_snippet', ''),
                'domain': metadata.get('domain', '')
            })
            
            # Stop when we have enough results
            if len(results) >= top_k:
                break
        
        return results


class ExperienceIndex:
    """
    Manages experience-specific indices with metadata.
    Builds indices from current experience data + replay buffer.
    """
    
    def __init__(
        self, 
        retriever_type: str,
        current_examples: List[Dict[str, Any]],
        replay_examples: List[Dict[str, Any]],
        experience_name: str,
        device: Optional[str] = None
    ):
        """
        Initialize and build the experience index.
        
        Args:
            retriever_type: Type of retriever to use
            current_examples: Examples from current experience
            replay_examples: Examples from replay buffer
            experience_name: Name of the current experience
            device: Device for GPU-based retrievers
        """
        self.retriever_type = retriever_type
        self.experience_name = experience_name
        self.device = device
        
        # Combine current and replay examples
        all_examples = current_examples + replay_examples
        
        # Extract prompts and build metadata
        corpus = []
        metadata_list = []
        # Track normalized prompt texts to avoid duplicates (text-level deduplication)
        added_normalized_texts = set()
        
        for idx, ex in enumerate(all_examples):
            prompt = ex.get("instruction", "").strip()
            if not prompt:
                continue
            
            # Skip if we've already added a prompt with the same normalized text
            normalized_prompt = normalize_text(prompt)
            if normalized_prompt in added_normalized_texts:
                continue
            
            # Get model card snippet from api_data
            model_card_snippet = ""
            api_data = ex.get("api_data", {})
            if isinstance(api_data, dict):
                model_card_snippet = api_data.get("description", "")
            elif isinstance(api_data, str):
                # Try to parse if it's a string representation
                try:
                    import json
                    api_data_dict = json.loads(api_data)
                    model_card_snippet = api_data_dict.get("description", "")
                except:
                    pass
            
            # Generate example_id if not present
            example_id = ex.get("example_id")
            if not example_id:
                model_name = ex.get("model_name", "")
                experience_id = ex.get("experience_id", experience_name)
                example_id = generate_example_id(prompt, model_name, experience_id, idx)
            
            # Get domain from entry (can be at top level or inside api_data)
            domain = ex.get("domain", "")
            if not domain and isinstance(api_data, dict):
                domain = api_data.get("domain", "")
            elif not domain and isinstance(api_data, str):
                try:
                    import json
                    api_data_dict = json.loads(api_data)
                    domain = api_data_dict.get("domain", "")
                except:
                    pass
            
            corpus.append(prompt)
            metadata_list.append({
                'example_id': example_id,
                'model_id': ex.get("model_name", ""),
                'experience_id': ex.get("experience_id", experience_name),
                'model_card_snippet': model_card_snippet,
                'domain': domain
            })
            added_normalized_texts.add(normalized_prompt)
        
        # Build retriever
        self.retriever = RetrieverWrapper(retriever_type, device)
        self.retriever.fit(corpus, metadata_list)
        
        print(f"  Built index for {experience_name}: {len(current_examples)} current + "
              f"{len(replay_examples)} replay = {len(corpus)} total examples")
    
    def retrieve(
        self, 
        query: str, 
        top_k: int, 
        exclude_example_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k similar prompts with metadata.
        
        Args:
            query: Query prompt string
            top_k: Number of results to return
            exclude_example_ids: List of example_ids to exclude (for self-masking)
            
        Returns:
            List of dicts with retrieval results
        """
        return self.retriever.retrieve_with_metadata(query, top_k, exclude_example_ids)

