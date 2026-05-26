from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import nn


def compute_router_fisher_information(
    *,
    lm_model: nn.Module,
    router_model: Optional[nn.Module],
    dataloader,
    routing_loss_fn: Callable[[nn.Module, Dict[str, Any], Any], Optional[torch.Tensor]],
    enable_hidden_states_fn: Callable[[nn.Module], None],
    num_samples: Optional[int] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Compute Fisher information for trainable router parameters."""
    if router_model is None:
        print("[RouterEWC] WARNING: No router model found; cannot compute Fisher.")
        return {}, {}

    print("\n[RouterEWC] Computing Fisher Information for router parameters...")

    gc_was_enabled = getattr(lm_model, "is_gradient_checkpointing", False)
    if gc_was_enabled:
        lm_model.gradient_checkpointing_disable()
        print("[RouterEWC] Gradient checkpointing disabled for Fisher computation")

    lm_model.eval()
    router_model.eval()
    enable_hidden_states_fn(lm_model)

    router_params = [(name, param) for name, param in router_model.named_parameters() if param.requires_grad]
    if not router_params:
        print("[RouterEWC] WARNING: No trainable router parameters found.")
        lm_model.train()
        router_model.train()
        return {}, {}

    print(f"[RouterEWC] Found {len(router_params)} router parameters for Fisher computation")
    fisher_dict = {name: torch.zeros_like(param.data) for name, param in router_params}

    sample_count = 0
    batch_count = 0
    total_samples = num_samples if num_samples is not None else float("inf")

    for batch_idx, batch in enumerate(dataloader):
        if sample_count >= total_samples:
            break

        inputs = {k: v.to(lm_model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        lm_model.zero_grad()
        router_model.zero_grad()

        with torch.enable_grad():
            outputs = lm_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                labels=inputs.get("labels"),
                output_hidden_states=True,
            )
            router_loss = routing_loss_fn(lm_model, inputs, outputs)

        if router_loss is None:
            continue

        router_loss.backward()

        for name, param in router_model.named_parameters():
            if name in fisher_dict and param.grad is not None:
                fisher_dict[name] += param.grad.data.pow(2)

        batch_size = inputs["input_ids"].size(0)
        sample_count += batch_size
        batch_count += 1

        del outputs, router_loss, inputs
        torch.cuda.empty_cache()

        if (batch_idx + 1) % 10 == 0:
            print(f"  [RouterEWC] Processed {sample_count} samples")

    if batch_count > 0:
        for name in fisher_dict:
            fisher_dict[name] /= float(batch_count)
            fisher_dict[name] = fisher_dict[name].cpu()

    print(
        f"[RouterEWC] Fisher computed: {sample_count} samples, {batch_count} batches. "
        "Matrices moved to CPU."
    )

    optimal_params = {
        name: param.data.clone().detach().cpu()
        for name, param in router_model.named_parameters()
        if param.requires_grad
    }

    lm_model.train()
    router_model.train()
    if gc_was_enabled:
        lm_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        print("[RouterEWC] Gradient checkpointing re-enabled")

    return fisher_dict, optimal_params
