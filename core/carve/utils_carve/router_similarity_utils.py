"""
Shared router similarity helpers.

This module intentionally contains only the utility functions that are
currently used across router training and replay candidate selection.
"""

import re
from typing import List, Optional, Union, Any

import torch
import torch.nn as nn


def canonicalize_domain(domain_str: Optional[str]) -> str:
    """
    Canonicalize a domain string for consistent comparison.

    Normalizes: lowercase, strip, collapse whitespace, handle None/"unknown".
    """
    if domain_str is None:
        return "unknown"

    domain_str = str(domain_str).lower().strip()
    domain_str = re.sub(r"\s+", " ", domain_str)

    if not domain_str:
        return "unknown"

    return domain_str


def build_taxonomy_soft_graph(
    batch_model_ids: List[str],
    batch_domains: List[str],
    alpha_domain: float = 0.3,
    device: Optional[Union[torch.device, str]] = None,
) -> torch.Tensor:
    """
    Build a taxonomy-based soft similarity graph.

    Graph weights:
    - 1.0 if same gold model_id
    - alpha_domain if same domain but different model_id
    - 0.0 otherwise

    Diagonal is set to -inf to exclude self-comparisons.
    """
    batch_size = len(batch_model_ids)

    if device is None:
        device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    batch_domains_canonical = [canonicalize_domain(d) for d in batch_domains]

    graph = torch.zeros(batch_size, batch_size, device=device)

    for i in range(batch_size):
        for j in range(batch_size):
            if i == j:
                continue
            if batch_model_ids[i] == batch_model_ids[j]:
                graph[i, j] = 1.0
            elif batch_domains_canonical[i] == batch_domains_canonical[j]:
                graph[i, j] = alpha_domain

    graph.fill_diagonal_(float("-inf"))
    return graph


def enable_hidden_states(model: nn.Module) -> None:
    """Enable output_hidden_states on model config(s) when available."""
    if hasattr(model, "config"):
        model.config.output_hidden_states = True
    if hasattr(model, "model") and hasattr(model.model, "config"):
        model.model.config.output_hidden_states = True
    if hasattr(model, "base_model"):
        if hasattr(model.base_model, "config"):
            model.base_model.config.output_hidden_states = True
        if hasattr(model.base_model, "model") and hasattr(model.base_model.model, "config"):
            model.base_model.model.config.output_hidden_states = True


def get_last_hidden_states(model_outputs: Any) -> Optional[torch.Tensor]:
    """Extract last-layer hidden states from model outputs."""
    if hasattr(model_outputs, "hidden_states") and model_outputs.hidden_states is not None:
        return model_outputs.hidden_states[-1]
    return None
