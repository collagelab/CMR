"""
Domain-aware batch samplers for semantic batching.

DomainBatchSampler yields batches where examples are mostly from the same domain,
maximizing overlap in semantic negative pools for routing loss.
"""

from typing import Iterator, List, Optional, Dict, Any
from collections import defaultdict, deque
import random
import torch
from torch.utils.data import Sampler, Dataset

from .router_exceptions import RouterTrainingError


class DomainBatchSampler(Sampler[List[int]]):
    """
    Sampler that yields domain-homogeneous batches.
    
    Strategy:
    1. Group dataset indices by domain
    2. For each batch, sample a domain (or multiple if domains_per_batch > 1)
    3. Fill batch from that domain, prioritizing current experience
    4. Optionally mix in replay examples from same domain
    
    This maximizes semantic negative overlap since examples in the batch
    share the same domain, allowing the router to learn fine-grained distinctions.
    
    Usage:
        sampler = DomainBatchSampler(
            dataset=train_dataset,
            batch_size=32,
            domains_per_batch=1,  # Currently only 1 is fully supported
            domain_key='domain',
            shuffle=True,
            drop_last=True,  # Recommended for stable training
        )
        
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
        )
    
    Note: Currently optimized for domains_per_batch=1 (pure domain batches).
    Mixed domain batching (domains_per_batch > 1) uses a simpler fallback strategy.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        domains_per_batch: int = 1,
        domain_key: str = 'domain',
        shuffle: bool = True,
        drop_last: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Initialize domain batch sampler.
        
        Args:
            dataset: Dataset with domain metadata
            batch_size: Batch size
            domains_per_batch: Number of domains per batch (1 = pure, >1 = mixed)
            domain_key: Key for domain field in dataset examples
            shuffle: Whether to shuffle indices within domains
            drop_last: Whether to drop incomplete batches
            seed: Random seed for reproducibility
        """
        # Validate inputs
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if domains_per_batch <= 0:
            raise ValueError(f"domains_per_batch must be positive, got {domains_per_batch}")
        # Check for required interface (duck typing) - works with both PyTorch and HuggingFace datasets
        if not (hasattr(dataset, '__len__') and hasattr(dataset, '__getitem__')):
            raise TypeError(f"dataset must support __len__ and __getitem__, got {type(dataset)}")
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.domains_per_batch = domains_per_batch
        self.domain_key = domain_key
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        if seed is not None:
            random.seed(seed)
        
        # Group indices by domain
        self.domain2indices: Dict[str, List[int]] = defaultdict(list)
        
        for idx in range(len(dataset)):
            example = dataset[idx]
            
            # Extract domain with fallbacks
            if isinstance(example, dict):
                domain = example.get(domain_key, 'unknown')
            else:
                # If dataset returns tuples or other types, try accessing as attribute
                domain = getattr(example, domain_key, 'unknown')
            
            # Canonicalize domain
            if isinstance(domain, str):
                domain = domain.strip().lower()
            else:
                domain = 'unknown'
            
            self.domain2indices[domain].append(idx)
        
        # Get list of domains
        self.domains = list(self.domain2indices.keys())
        
        # Shuffle indices within each domain if requested
        if self.shuffle:
            for domain in self.domains:
                random.shuffle(self.domain2indices[domain])
        
        # Precompute number of batches
        self._length = self._compute_length()
    
    def _compute_length(self) -> int:
        """Compute number of batches this sampler will yield."""
        total_samples = len(self.dataset)
        
        if self.drop_last:
            return total_samples // self.batch_size
        else:
            return (total_samples + self.batch_size - 1) // self.batch_size
    
    def __iter__(self) -> Iterator[List[int]]:
        """
        Iterate over batches using robust round-robin over active domains.
        
        Strategy:
        - Create a deque for each domain with its shuffled indices
        - Maintain an active_domains queue containing domains with remaining samples
        - Round-robin: pop domain from left, take samples, re-add to right if not exhausted
        - Naturally handles exhausted domains (they drop out of the queue)
        - Guarantees all samples are yielded exactly once per epoch
        """
        # Create deques of indices for each domain (shuffled once per epoch)
        domain_pools: Dict[str, deque] = {}
        for domain, indices in self.domain2indices.items():
            indices_copy = indices.copy()
            if self.shuffle:
                random.shuffle(indices_copy)
            domain_pools[domain] = deque(indices_copy)
        
        # Initialize active domains queue (domains with remaining samples)
        # Shuffle domain order for initial round-robin
        domain_order = self.domains.copy()
        if self.shuffle:
            random.shuffle(domain_order)
        active_domains = deque([d for d in domain_order if len(domain_pools[d]) > 0])
        
        batches_yielded = 0
        
        # Round-robin over active domains
        while active_domains:
            # Pop domain from left (FIFO)
            current_domain = active_domains.popleft()
            pool = domain_pools[current_domain]
            
            # Take up to batch_size samples from this domain
            batch = []
            while len(batch) < self.batch_size and pool:
                batch.append(pool.popleft())
            
            # Decide whether to yield this batch
            if len(batch) == self.batch_size:
                # Full batch - always yield
                batches_yielded += 1
                yield batch
                
                # Re-add domain to queue if it still has samples
                if pool:
                    active_domains.append(current_domain)
                    
            elif not self.drop_last and len(batch) > 0:
                # Partial batch and drop_last=False - yield it
                batches_yielded += 1
                yield batch
                
                # Don't re-add domain (it's exhausted)
                
            else:
                # Partial batch but drop_last=True - drop it and don't re-add domain
                pass
        
        # Diagnostic: Check if we yielded the expected number of batches
        expected = self._length
        if batches_yielded != expected:
            # Only warn if significant mismatch (off by more than 1)
            # This can happen with drop_last=False when domains have different sizes
            if abs(batches_yielded - expected) > 1:
                import warnings
                warnings.warn(
                    f"DomainBatchSampler: Yielded {batches_yielded} batches, expected {expected}. "
                    f"This may indicate an issue with batch computation or domain distribution.",
                    UserWarning,
                    stacklevel=2
                )
    
    def __len__(self) -> int:
        """Number of batches."""
        return self._length


class MixedDomainBatchSampler(Sampler[List[int]]):
    """
    Sampler that yields mostly-homogeneous batches with optional replay mixing.
    
    This is similar to DomainBatchSampler but supports mixing replay examples
    from the same domain into each batch.
    
    Strategy:
    1. Sample a domain
    2. Fill most of batch from current experience examples in that domain
    3. Top up with replay examples from same domain (if mix_replay=True)
    4. Fallback to other domains if domain is sparse
    
    Usage:
        sampler = MixedDomainBatchSampler(
            dataset=combined_dataset,
            batch_size=32,
            replay_mask=[0,0,1,1,...],  # 1 = replay, 0 = current
            mix_replay=True,
            replay_fraction=0.25,
        )
    """
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        replay_mask: Optional[List[int]] = None,
        mix_replay: bool = True,
        replay_fraction: float = 0.25,
        domain_key: str = 'domain',
        shuffle: bool = True,
        drop_last: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Initialize mixed domain batch sampler.
        
        Args:
            dataset: Combined dataset (current + replay)
            batch_size: Batch size
            replay_mask: Binary mask indicating replay examples (1 = replay, 0 = current)
            mix_replay: Whether to mix replay into batches
            replay_fraction: Target fraction of replay in each batch
            domain_key: Key for domain field
            shuffle: Whether to shuffle
            drop_last: Whether to drop incomplete batches
            seed: Random seed
        """
        # Validate inputs
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if not (0.0 <= replay_fraction <= 1.0):
            raise ValueError(
                f"replay_fraction must be in [0.0, 1.0], got {replay_fraction}"
            )
        # Check for required interface (duck typing) - works with both PyTorch and HuggingFace datasets
        if not (hasattr(dataset, '__len__') and hasattr(dataset, '__getitem__')):
            raise TypeError(f"dataset must support __len__ and __getitem__, got {type(dataset)}")
        if replay_mask is not None and len(replay_mask) != len(dataset):
            raise ValueError(
                f"replay_mask length ({len(replay_mask)}) must match dataset length ({len(dataset)})"
            )
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.replay_mask = replay_mask or [0] * len(dataset)
        self.mix_replay = mix_replay
        self.replay_fraction = replay_fraction
        self.domain_key = domain_key
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        if seed is not None:
            random.seed(seed)
        
        # Group indices by (domain, is_replay)
        self.domain_current_indices: Dict[str, List[int]] = defaultdict(list)
        self.domain_replay_indices: Dict[str, List[int]] = defaultdict(list)
        
        for idx in range(len(dataset)):
            example = dataset[idx]
            is_replay = self.replay_mask[idx] == 1
            
            # Extract domain
            if isinstance(example, dict):
                domain = example.get(domain_key, 'unknown')
            else:
                domain = getattr(example, domain_key, 'unknown')
            
            # Canonicalize
            if isinstance(domain, str):
                domain = domain.strip().lower()
            else:
                domain = 'unknown'
            
            if is_replay:
                self.domain_replay_indices[domain].append(idx)
            else:
                self.domain_current_indices[domain].append(idx)
        
        self.domains = list(set(
            list(self.domain_current_indices.keys()) +
            list(self.domain_replay_indices.keys())
        ))
        
        # Shuffle indices
        if self.shuffle:
            for domain in self.domains:
                if domain in self.domain_current_indices:
                    random.shuffle(self.domain_current_indices[domain])
                if domain in self.domain_replay_indices:
                    random.shuffle(self.domain_replay_indices[domain])
        
        self._length = self._compute_length()
    
    def _compute_length(self) -> int:
        """Compute number of batches."""
        total = len([idx for idx, is_replay in enumerate(self.replay_mask) if not is_replay])
        
        if self.drop_last:
            return total // self.batch_size
        else:
            return (total + self.batch_size - 1) // self.batch_size
    
    def __iter__(self) -> Iterator[List[int]]:
        """Iterate over batches."""
        # Create working pools
        current_pool = {
            domain: indices.copy()
            for domain, indices in self.domain_current_indices.items()
        }
        replay_pool = {
            domain: indices.copy()
            for domain, indices in self.domain_replay_indices.items()
        }
        
        domain_order = self.domains.copy()
        if self.shuffle:
            random.shuffle(domain_order)
        
        domain_idx = 0
        
        while True:
            batch = []
            
            # Select domain
            current_domain = domain_order[domain_idx % len(domain_order)]
            domain_idx += 1
            
            # Determine split between current and replay
            if self.mix_replay:
                num_replay = int(self.batch_size * self.replay_fraction)
                num_current = self.batch_size - num_replay
            else:
                num_current = self.batch_size
                num_replay = 0
            
            # Fill from current experience
            if current_domain in current_pool:
                available = current_pool[current_domain]
                num_to_take = min(num_current, len(available))
                batch.extend(available[:num_to_take])
                current_pool[current_domain] = available[num_to_take:]
            
            # Fill from replay (same domain)
            if self.mix_replay and current_domain in replay_pool:
                available = replay_pool[current_domain]
                slots_remaining = self.batch_size - len(batch)
                num_to_take = min(num_replay, len(available), slots_remaining)
                batch.extend(available[:num_to_take])
                replay_pool[current_domain] = available[num_to_take:]
            
            # Fallback: fill from any remaining current examples
            if len(batch) < self.batch_size:
                for domain in self.domains:
                    if domain in current_pool and current_pool[domain]:
                        available = current_pool[domain]
                        slots_remaining = self.batch_size - len(batch)
                        num_to_take = min(slots_remaining, len(available))
                        batch.extend(available[:num_to_take])
                        current_pool[domain] = available[num_to_take:]
                        
                        if len(batch) >= self.batch_size:
                            break
            
            # Check stop conditions
            if len(batch) < self.batch_size and self.drop_last:
                break
            
            if not batch:
                break
            
            yield batch
    
    def __len__(self) -> int:
        """Number of batches."""
        return self._length

