"""
Model Registry for stable model name → ID mapping with metadata.

The registry collects all unique model names from training data, replay buffer,
and candidate pools, assigns stable integer IDs, and stores metadata like domain
and family information.
"""

from typing import Dict, List, Optional, Set, Any
from collections import defaultdict
import json


def normalize_model_name(model_name: str) -> str:
    """
    Normalize model name for consistent matching (case-insensitive).
    
    Args:
        model_name: Raw model name string
    
    Returns:
        Normalized model name (lowercased, stripped)
    """
    if not isinstance(model_name, str):
        return str(model_name) if model_name is not None else ""
    
    return model_name.strip().lower()


def normalize_domain(domain: str) -> str:
    """
    Normalize domain string for consistent matching.
    
    Args:
        domain: Raw domain string
    
    Returns:
        Normalized domain string (lowercased, stripped, collapsed spaces)
    """
    if not isinstance(domain, str):
        return "unknown"
    
    domain = domain.strip().lower()
    domain = " ".join(domain.split())
    return domain if domain else "unknown"


def extract_parent_group(domain: str) -> str:
    """
    Extract parent group from domain string using prefix matching.
    
    This function derives a high-level domain category from specific domain strings
    to enable semantic pool expansion beyond exact domain matches.
    
    Args:
        domain: Normalized domain string (e.g., "natural language processing text2text generation")
    
    Returns:
        Parent group string (e.g., "natural language processing")
    
    Examples:
        >>> extract_parent_group("natural language processing text2text generation")
        "natural language processing"
        >>> extract_parent_group("computer vision object detection")
        "computer vision"
        >>> extract_parent_group("unknown")
        "other"
    """
    domain = normalize_domain(domain)
    
    prefixes = [
        "natural language processing",
        "computer vision",
        "audio",
        "multimodal",
        "tabular",
        "reinforcement learning",
        "time series",
        "graph",
    ]
    
    for prefix in prefixes:
        if domain.startswith(prefix):
            return prefix
    
    tokens = domain.split()
    if tokens and tokens[0] not in ["unknown", ""]:
        return tokens[0]
    
    return "other"


class ModelRegistry:
    """
    Centralized registry mapping model names to stable integer IDs with metadata.
    
    Provides:
    - model2idx, idx2model mappings
    - domain2models grouping for semantic sampling
    - get_neighbors() using taxonomy structure (reuses X-CLR graph)
    
    Usage:
        registry = ModelRegistry.from_examples(
            train_examples=train_data,
            replay_examples=replay_buffer,
            raw_prompts=prompts
        )
        model_idx = registry.model2idx["model_name"]
        semantic_pool = registry.domain2models["cv"]
    """
    
    def __init__(self):
        self.model2idx: Dict[str, int] = {}
        self.idx2model: Dict[int, str] = {}
        self.metadata: Dict[int, Dict[str, Any]] = {}  # idx -> {domain, family, model_text, ...}
        self.domain2models: Dict[str, List[int]] = defaultdict(list)
        self.family2models: Dict[str, List[int]] = defaultdict(list)
        self._num_models: int = 0
        self.parent_group2models: Dict[str, List[int]] = defaultdict(list)  # parent_group -> [model_idx]
        self.domain2parent_group: Dict[str, str] = {}  # domain -> parent_group
        self._semantic_pool_cache: Dict[tuple, List[int]] = {}
    
    @classmethod
    def from_examples(
        cls,
        train_examples: Optional[List[Dict[str, Any]]] = None,
        replay_examples: Optional[List[Dict[str, Any]]] = None,
        raw_prompts: Optional[List[Dict[str, Any]]] = None,
        model_name_key: str = "model_name",
        domain_key: str = "domain",
        family_key: Optional[str] = None,
    ) -> "ModelRegistry":
        """
        Build registry from training data, replay buffer, and prompts.
        
        Args:
            train_examples: Training dataset examples
            replay_examples: Replay buffer examples
            raw_prompts: Raw prompts with metadata
            model_name_key: Key for model name in examples
            domain_key: Key for domain in examples
            family_key: Optional key for model family
        
        Returns:
            ModelRegistry instance
        """
        registry = cls()
        all_examples = []
        if train_examples:
            all_examples.extend(train_examples)
        if replay_examples:
            all_examples.extend(replay_examples)
        if raw_prompts:
            all_examples.extend(raw_prompts)
        
        model_metadata: Dict[str, Dict[str, Any]] = {}
        
        for example in all_examples:
            model_name = example.get(model_name_key)
            if not model_name:
                model_name = example.get("model_id") or example.get("model")
            if not model_name or not isinstance(model_name, str):
                continue
            
            if model_name in model_metadata:
                continue
            
            domain = example.get(domain_key, "unknown")
            if not domain:
                api_data = example.get("api_data", {})
                if isinstance(api_data, dict):
                    domain = api_data.get("domain", "unknown")
            
            domain = normalize_domain(domain)
            
            family = None
            if family_key:
                family = example.get(family_key)
            
            model_text = model_name
            api_data = example.get("api_data", {})
            card_text = None
            if isinstance(api_data, dict):
                card_text = (
                    api_data.get("description")
                    or api_data.get("model_card")
                    or api_data.get("model_card_snippet")
                )
            if not card_text:
                card_text = (
                    example.get("model_card")
                    or example.get("model_card_snippet")
                    or example.get("description")
                )
            
            model_metadata[model_name] = {
                "domain": domain,
                "family": family,
                "model_text": model_text,
                "card_text": str(card_text) if card_text else None,
            }
        
        sorted_models = sorted(model_metadata.keys())
        for idx, model_name in enumerate(sorted_models):
            registry.model2idx[model_name] = idx
            registry.idx2model[idx] = model_name
            registry.metadata[idx] = model_metadata[model_name]
            
            domain = model_metadata[model_name]["domain"]
            registry.domain2models[domain].append(idx)
            
            family = model_metadata[model_name].get("family")
            if family:
                registry.family2models[family].append(idx)
        
        registry._num_models = len(sorted_models)
        registry._build_parent_group_mappings()
        
        return registry
    
    def get_neighbors(
        self,
        model_idx: int,
        k: int = 5,
        strategy: str = "domain_then_family"
    ) -> List[int]:
        """
        Get k nearest neighbors for a model based on taxonomy.
        
        Neighbors are defined by:
        1. Same domain (priority)
        2. Same family (if available)
        3. Random models as fallback
        
        Args:
            model_idx: Model index to find neighbors for
            k: Number of neighbors to return
            strategy: "domain_then_family" or "family_then_domain"
        
        Returns:
            List of neighbor model indices (excluding model_idx itself)
        """
        if model_idx not in self.idx2model:
            return []
        
        metadata = self.metadata[model_idx]
        domain = metadata["domain"]
        family = metadata.get("family")
        
        neighbors = []
        
        if strategy == "domain_then_family":
            domain_models = [idx for idx in self.domain2models[domain] if idx != model_idx]
            neighbors.extend(domain_models[:k])
            
            if family and len(neighbors) < k:
                family_models = [idx for idx in self.family2models[family] 
                                if idx != model_idx and idx not in neighbors]
                neighbors.extend(family_models[:k - len(neighbors)])
        
        elif strategy == "family_then_domain":
            if family:
                family_models = [idx for idx in self.family2models[family] if idx != model_idx]
                neighbors.extend(family_models[:k])
            
            if len(neighbors) < k:
                domain_models = [idx for idx in self.domain2models[domain] 
                                if idx != model_idx and idx not in neighbors]
                neighbors.extend(domain_models[:k - len(neighbors)])
        
        if len(neighbors) < k:
            all_other_models = [idx for idx in range(self._num_models) 
                               if idx != model_idx and idx not in neighbors]
            import random
            random.shuffle(all_other_models)
            neighbors.extend(all_other_models[:k - len(neighbors)])
        
        return neighbors[:k]
    
    def _build_parent_group_mappings(self):
        """
        Build parent_group2models and domain2parent_group mappings.
        
        Called internally after models are registered.
        """
        self.parent_group2models.clear()
        self.domain2parent_group.clear()
        
        for domain, model_indices in self.domain2models.items():
            parent_group = extract_parent_group(domain)
            self.domain2parent_group[domain] = parent_group
            self.parent_group2models[parent_group].extend(model_indices)
    
    def get_semantic_pool(
        self,
        domain: str,
        mode: str = "parent_group",
        depth: int = 1,
        max_domains: Optional[int] = None,
        exclude: Optional[Set[int]] = None,
    ) -> List[int]:
        """
        Get expanded semantic pool for a domain using related domains.
        
        This is the core function for Option B: expands semantic negatives beyond
        exact domain to handle sparse domains.
        
        Args:
            domain: Target domain (will be normalized)
            mode: Semantic pool mode
                - "domain_only": Only exact domain (original behavior)
                - "parent_group": Same domain + domains sharing parent group
                - "taxonomy_graph": Use explicit taxonomy graph (if available)
            depth: Graph traversal depth (only for taxonomy_graph mode)
            max_domains: Maximum number of related domains to include (None = all)
            exclude: Set of model indices to exclude from pool
        
        Returns:
            List of model indices from semantic pool (deduplicated)
        
        Examples:
            >>> # domain_only mode (original behavior)
            >>> pool = registry.get_semantic_pool("computer vision image classification", mode="domain_only")
            >>> # Returns only models from exact domain
            
            >>> # parent_group mode (Option B)
            >>> pool = registry.get_semantic_pool("computer vision image classification", mode="parent_group")
            >>> # Returns models from all "computer vision *" domains
        """
        domain = normalize_domain(domain)
        exclude = exclude or set()
        
        cache_key = (domain, mode, depth, max_domains, len(exclude) == 0)
        if len(exclude) == 0 and cache_key in self._semantic_pool_cache:
            return self._semantic_pool_cache[cache_key]
        
        semantic_pool: List[int] = []
        
        if mode == "domain_only":
            semantic_pool = list(self.domain2models.get(domain, []))
        
        elif mode == "parent_group":
            parent_group = self.domain2parent_group.get(domain, extract_parent_group(domain))
            
            related_domains = [
                d for d, pg in self.domain2parent_group.items()
                if pg == parent_group
            ]
            
            if max_domains is not None and len(related_domains) > max_domains:
                if domain in related_domains:
                    other_domains = [d for d in related_domains if d != domain]
                    import random
                    sampled_others = random.sample(other_domains, min(max_domains - 1, len(other_domains)))
                    related_domains = [domain] + sampled_others
                else:
                    import random
                    related_domains = random.sample(related_domains, max_domains)
            
            for related_domain in related_domains:
                semantic_pool.extend(self.domain2models.get(related_domain, []))
            
            semantic_pool = list(dict.fromkeys(semantic_pool))
        
        elif mode == "taxonomy_graph":
            parent_group = self.domain2parent_group.get(domain, extract_parent_group(domain))
            related_domains = [
                d for d, pg in self.domain2parent_group.items()
                if pg == parent_group
            ]
            
            if max_domains is not None and len(related_domains) > max_domains:
                if domain in related_domains:
                    other_domains = [d for d in related_domains if d != domain]
                    import random
                    sampled_others = random.sample(other_domains, min(max_domains - 1, len(other_domains)))
                    related_domains = [domain] + sampled_others
                else:
                    import random
                    related_domains = random.sample(related_domains, max_domains)
            
            for related_domain in related_domains:
                semantic_pool.extend(self.domain2models.get(related_domain, []))
            
            semantic_pool = list(dict.fromkeys(semantic_pool))
        
        else:
            raise ValueError(f"Unknown semantic_pool_mode: {mode}. Use 'domain_only', 'parent_group', or 'taxonomy_graph'")
        
        if exclude:
            semantic_pool = [idx for idx in semantic_pool if idx not in exclude]
        
        if len(exclude) == 0:
            self._semantic_pool_cache[cache_key] = semantic_pool
        
        return semantic_pool
    
    def get_parent_group_stats(self) -> Dict[str, int]:
        """
        Get statistics about parent groups for diagnostics.
        
        Returns:
            Dict mapping parent_group -> model_count
        """
        return {
            parent_group: len(model_indices)
            for parent_group, model_indices in self.parent_group2models.items()
        }
    
    def get_domain_models(self, domain: str, exclude: Optional[Set[int]] = None) -> List[int]:
        """
        Get all models in a domain, optionally excluding some.
        
        Args:
            domain: Domain string (will be canonicalized)
            exclude: Optional set of model indices to exclude
        
        Returns:
            List of model indices in the domain
        """
        domain = domain.strip().lower() if isinstance(domain, str) else "unknown"
        
        models = self.domain2models[domain]
        
        if exclude:
            models = [idx for idx in models if idx not in exclude]
        
        return models
    
    def get_all_domains(self) -> List[str]:
        """Get list of all unique domains."""
        return list(self.domain2models.keys())
    
    def get_other_domains(self, exclude_domain: str) -> List[str]:
        """Get all domains except the specified one."""
        return [d for d in self.domain2models.keys() if d != exclude_domain]
    
    def __len__(self) -> int:
        """Number of unique models in registry."""
        return self._num_models
    
    def save(self, path: str):
        """
        Save registry to JSON file for persistence across experiences.
        
        IMPORTANT: In continual learning, the registry should be built once
        at the start and persisted, not rebuilt from scratch each experience.
        """
        data = {
            "model2idx": self.model2idx,
            "idx2model": {str(k): v for k, v in self.idx2model.items()},  # JSON keys must be strings
            "metadata": {str(k): v for k, v in self.metadata.items()},
            "num_models": self._num_models,
            "domain2models": {domain: indices for domain, indices in self.domain2models.items()},
            "graph": getattr(self, "graph", None),
            "parent_group2models": {pg: indices for pg, indices in self.parent_group2models.items()},
            "domain2parent_group": self.domain2parent_group,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"✓ Saved ModelRegistry to {path}")
    
    @classmethod
    def load(cls, path: str) -> "ModelRegistry":
        """Load registry from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        
        registry = cls()
        registry.model2idx = data["model2idx"]
        registry.idx2model = {int(k): v for k, v in data["idx2model"].items()}
        registry.metadata = {int(k): v for k, v in data["metadata"].items()}
        registry._num_models = data["num_models"]
        
        if "domain2models" in data:
            registry.domain2models = defaultdict(list, {
                domain: indices for domain, indices in data["domain2models"].items()
            })
        else:
            for idx, metadata in registry.metadata.items():
                domain = metadata["domain"]
                registry.domain2models[domain].append(idx)
        
        if "graph" in data:
            registry.graph = data["graph"]
        
        for idx, metadata in registry.metadata.items():
            family = metadata.get("family")
            if family:
                registry.family2models[family].append(idx)
        
        if "parent_group2models" in data and "domain2parent_group" in data:
            registry.parent_group2models = defaultdict(list, {
                pg: indices for pg, indices in data["parent_group2models"].items()
            })
            registry.domain2parent_group = data["domain2parent_group"]
        else:
            registry._build_parent_group_mappings()
        
        print(f"✓ Loaded ModelRegistry from {path}")
        print(f"  {registry._num_models} models, {len(registry.domain2models)} domains, {len(registry.parent_group2models)} parent groups")
        
        return registry
    
    def add_model(
        self,
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Add a single model to the registry with a new ID.
        
        Args:
            model_name: Model name (will be normalized for lookup but stored as-is)
            metadata: Optional metadata dict with keys like 'domain', 'family', 'model_text'
        
        Returns:
            The assigned model index (ID)
        
        Note:
            This method does NOT check for duplicates. Use extend_from_examples for bulk operations
            that handle duplicate detection.
        """
        normalized = normalize_model_name(model_name)
        
        for existing_name in self.model2idx.keys():
            if normalize_model_name(existing_name) == normalized:
                return self.model2idx[existing_name]
        
        new_idx = self._num_models
        
        self.model2idx[model_name] = new_idx
        self.idx2model[new_idx] = model_name
    
        if metadata is None:
            metadata = {}
        self.metadata[new_idx] = {
            "domain": metadata.get("domain", "unknown"),
            "family": metadata.get("family"),
            "model_text": metadata.get("model_text", model_name),
        }
        
        domain = normalize_domain(self.metadata[new_idx]["domain"])
        self.domain2models[domain].append(new_idx)
        
        family = self.metadata[new_idx].get("family")
        if family:
            self.family2models[family].append(new_idx)
        
        self._num_models += 1
        
        return new_idx
    
    def extend_from_examples(
        self,
        examples: List[Dict[str, Any]],
        model_name_key: str = "model_name",
        domain_key: str = "domain",
        family_key: Optional[str] = None,
    ) -> int:
        """
        Extend registry by adding any models from examples that are not already present.
        
        This method preserves existing model IDs and only appends new models.
        It uses case-insensitive matching to detect duplicates.
        
        Args:
            examples: List of example dicts containing model names
            model_name_key: Key for model name in examples
            domain_key: Key for domain in examples
            family_key: Optional key for model family
        
        Returns:
            Number of new models added
        """
        existing_normalized = {
            normalize_model_name(name): name 
            for name in self.model2idx.keys()
        }
        
        new_model_metadata: Dict[str, Dict[str, Any]] = {}
        
        for example in examples:
            model_name = example.get(model_name_key)
            if not model_name:
                model_name = example.get("model_id") or example.get("model")
            if not model_name or not isinstance(model_name, str):
                continue
            
            normalized = normalize_model_name(model_name)
            if normalized in existing_normalized:
                continue
            
            if model_name in new_model_metadata:
                continue
            
            domain = example.get(domain_key, "unknown")
            if not domain:
                api_data = example.get("api_data", {})
                if isinstance(api_data, dict):
                    domain = api_data.get("domain", "unknown")
            
            domain = normalize_domain(domain)
            
            family = None
            if family_key:
                family = example.get(family_key)
            
            model_text = model_name
            
            new_model_metadata[model_name] = {
                "domain": domain,
                "family": family,
                "model_text": model_text,
            }
        
        sorted_new_models = sorted(new_model_metadata.keys())
        num_added = 0
        
        for model_name in sorted_new_models:
            metadata = new_model_metadata[model_name]
            self.add_model(model_name, metadata)
            num_added += 1
        
        if num_added > 0:
            self._build_parent_group_mappings()
            self._semantic_pool_cache.clear()
        
        return num_added
    
    def __repr__(self) -> str:
        return (f"ModelRegistry(num_models={self._num_models}, "
                f"num_domains={len(self.domain2models)})")

