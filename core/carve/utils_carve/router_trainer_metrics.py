from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class LossAccum:
    """Accumulates a running sum, count, and weighted sum for one loss term."""

    total: float = 0.0
    count: int = 0
    weighted: float = 0.0

    def update(self, raw: float, weighted: float = 0.0) -> None:
        self.total += raw
        self.count += 1
        self.weighted += weighted

    def average(self) -> Optional[float]:
        return self.total / self.count if self.count else None

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0
        self.weighted = 0.0


def get_router_metrics(trainer) -> Dict[str, float]:
    """Compute aggregated router metrics from trainer state."""
    metrics: Dict[str, float] = {}
    if trainer._router_accum.count > 0:
        avg_router = trainer._router_accum.total / trainer._router_accum.count
        avg_supervised = trainer._supervised_accum.total / max(1, trainer._supervised_accum.count)
        metrics["avg_router_loss"] = avg_router
        metrics["avg_supervised_loss"] = avg_supervised
        if avg_supervised > 0:
            metrics["router_to_supervised_ratio"] = avg_router / avg_supervised

    if trainer._router_graph_accum.count > 0:
        avg_graph = trainer._router_graph_accum.total / trainer._router_graph_accum.count
        metrics["avg_graph_loss"] = avg_graph

    if trainer._router_ewc_accum.count > 0:
        avg_ewc = trainer._router_ewc_accum.total / trainer._router_ewc_accum.count
        metrics["avg_router_ewc_loss"] = avg_ewc

    if hasattr(trainer, "_router_accuracy_sum") and trainer._router_accuracy_count > 0:
        for key, value_sum in trainer._router_accuracy_sum.items():
            avg_value = value_sum / trainer._router_accuracy_count
            metrics[f"avg_{key}"] = avg_value

    if trainer._router_hard_miner:
        stats = trainer._router_hard_miner.get_stats()
        metrics.update(
            {
                "hard_mining_updates": stats["num_updates"],
                "hard_mining_examples_processed": stats["num_examples_processed"],
                "hard_mining_cache_size": stats["cache_size"],
            }
        )
    return metrics


def reset_consistency_metrics(trainer) -> None:
    """Reset per-epoch metric tracking fields on trainer."""
    trainer._consistency_accum.reset()
    trainer._supervised_accum.reset()
    trainer._contrastive_accum.reset()
    trainer._router_accum.reset()
    trainer._router_graph_accum.reset()
    trainer._router_anchor_accum.reset()
    trainer._router_proj_anchor_accum.reset()
    trainer._router_ewc_accum.reset()
    trainer._contrastive_anchors_used = 0
    trainer._contrastive_negatives_used = 0
    if hasattr(trainer, "_router_accuracy_sum"):
        for key in trainer._router_accuracy_sum.keys():
            trainer._router_accuracy_sum[key] = 0.0
        trainer._router_accuracy_count = 0
    trainer._neighbor_domain_stats = defaultdict(lambda: {"same": 0, "different": 0})
