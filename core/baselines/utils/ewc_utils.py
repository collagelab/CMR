"""
Elastic Weight Consolidation (EWC) utilities for continual learning with LoRA.

Supports both 'separate' and 'online' modes, matching Avalanche's EWCPlugin logic.

Fixes vs original ewc_utils.py:
  1. model.train() is restored after Fisher computation.
  2. Gradient checkpointing is disabled/re-enabled around Fisher computation.
  3. Lambda scaling uses ewc_lambda * loss (no /2), consistent with Avalanche.
  4. Online mode consolidates Fisher across tasks with decay, preventing
     constraint conflicts that caused catastrophic forgetting with separate mode.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch


class EWCRegularizer:
    """
    Elastic Weight Consolidation (EWC) regularizer for LoRA parameters.

    Two modes:
      - 'separate': keeps an independent Fisher + optimal-params snapshot for
        every previous experience and sums all penalties. Memory grows with
        the number of tasks.
      - 'online': maintains a single consolidated Fisher that is updated with
        exponential decay at each new experience, matching Avalanche's online
        EWC. Memory stays constant regardless of task count.

    For continual learning with many tasks, 'online' mode is strongly preferred.
    """

    def __init__(
        self,
        ewc_lambda: float = 1000.0,
        mode: str = "online",
        decay_factor: float = 0.9,
    ):
        """
        Args:
            ewc_lambda: Regularization strength. Larger values preserve old
                tasks more strongly.  Start around 1e5 for LoRA fine-tuning.
            mode: 'online' (recommended) or 'separate'.
            decay_factor: Only used in online mode. Controls how quickly old
                task importances are discounted when a new task is added.
                Typical range: 0.7–0.95.
        """
        assert mode in ("separate", "online"), "mode must be 'separate' or 'online'"
        if mode == "online":
            assert 0.0 < decay_factor <= 1.0, "decay_factor must be in (0, 1]"

        self.ewc_lambda = ewc_lambda
        self.mode = mode
        self.decay_factor = decay_factor

        # --- online mode state (single consolidated snapshot) ---
        self.consolidated_fisher: Dict[str, torch.Tensor] = {}
        self.consolidated_params: Dict[str, torch.Tensor] = {}

        # --- separate mode state (one snapshot per experience) ---
        self.fisher_dict_list: List[Dict[str, torch.Tensor]] = []
        self.optimal_params_list: List[Dict[str, torch.Tensor]] = []

        # Number of experiences added so far (used for logging)
        self._num_experiences: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_experience(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        num_samples: Optional[int] = None,
    ) -> None:
        """
        Call this *after* training on each experience.

        Computes the Fisher Information Matrix for the current task and stores
        the current LoRA parameters as the optimal reference point.

        Args:
            model: The trained model with LoRA adapters.
            dataloader: DataLoader for the current task's training data.
            num_samples: Cap on the number of samples used for Fisher
                computation. None = use all data.
        """
        print(f"\n[EWC] Computing Fisher Information for experience {self._num_experiences + 1}...")

        fisher_dict = self.compute_fisher_information(model, dataloader, num_samples)
        optimal_params = self._get_optimal_params(model)

        if self.mode == "online":
            self._update_consolidated(fisher_dict, optimal_params)
        else:
            self.fisher_dict_list.append(fisher_dict)
            self.optimal_params_list.append(optimal_params)

        self._num_experiences += 1
        print(f"[EWC] Experience {self._num_experiences} added (mode={self.mode})")

    def compute_ewc_loss(self, model: torch.nn.Module) -> torch.Tensor:
        """
        Compute the EWC regularization penalty.

        Loss = ewc_lambda * Σ_i Σ_j  F_ij * (θ_j - θ*_ij)²

        Returns a scalar tensor on the same device as the model.
        """
        device = next(
            (p.device for p in model.parameters() if p.requires_grad),
            torch.device("cpu"),
        )

        if self.mode == "online":
            return self._ewc_loss_online(model, device)
        else:
            return self._ewc_loss_separate(model, device)

    # ------------------------------------------------------------------
    # Fisher computation
    # ------------------------------------------------------------------

    def compute_fisher_information(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        num_samples: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the empirical Fisher Information Matrix for all LoRA parameters.

        The Fisher is approximated as the average of squared gradients over the
        training data, matching Avalanche's compute_importances implementation.

        Args:
            model: Model with LoRA adapters.
            dataloader: DataLoader for the task's training data.
            num_samples: If set, stop after this many samples.

        Returns:
            Dict mapping parameter names → Fisher tensors (on CPU).
        """
        # --- 1. Temporarily disable gradient checkpointing (can interfere with
        #        backward in eval mode) ---
        gc_was_enabled = getattr(model, "is_gradient_checkpointing", False)
        if gc_was_enabled:
            model.gradient_checkpointing_disable()
            print("[EWC] Gradient checkpointing disabled for Fisher computation")

        # --- 2. Switch to eval mode (but keep RNNs in train mode on CUDA,
        #        matching Avalanche's workaround) ---
        model.eval()
        if next(model.parameters()).device.type == "cuda":
            for module in model.modules():
                if isinstance(module, torch.nn.RNNBase):
                    module.train()

        # --- 3. Identify LoRA parameters and initialise accumulators ---
        lora_params = [
            (name, param)
            for name, param in model.named_parameters()
            if param.requires_grad and "lora" in name.lower()
        ]
        if not lora_params:
            print("[EWC] WARNING: No LoRA parameters found with requires_grad=True")

        print(f"[EWC] Found {len(lora_params)} LoRA parameters for Fisher computation")

        fisher_dict: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(param.data) for name, param in lora_params
        }

        # --- 4. Accumulate squared gradients ---
        sample_count = 0
        batch_count = 0
        total_samples = num_samples if num_samples is not None else float("inf")

        for batch_idx, batch in enumerate(dataloader):
            if sample_count >= total_samples:
                break

            # Support both dict and tuple batches
            if isinstance(batch, dict):
                batch_on_device = {
                    k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                input_ids = batch_on_device["input_ids"]
                attention_mask = batch_on_device.get("attention_mask", None)
                labels = batch_on_device["labels"]
            else:
                input_ids = batch[0].to(model.device)
                attention_mask = batch[1].to(model.device) if len(batch) > 1 else None
                labels = batch[2].to(model.device) if len(batch) > 2 else input_ids

            model.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()

            # Accumulate squared gradients (Fisher approximation)
            for name, param in model.named_parameters():
                if name in fisher_dict and param.grad is not None:
                    fisher_dict[name] += param.grad.data.pow(2)

            batch_size = input_ids.size(0)
            sample_count += batch_size
            batch_count += 1

            # Free intermediates immediately to avoid GPU memory accumulation
            del outputs, loss, input_ids, attention_mask, labels
            if isinstance(batch, dict):
                del batch_on_device
            torch.cuda.empty_cache()

            if (batch_idx + 1) % 10 == 0:
                print(f"  [EWC] Processed {sample_count} samples")

        # --- 5. Average over batches (matching Avalanche), move to CPU ---
        if batch_count > 0:
            for name in fisher_dict:
                fisher_dict[name] /= float(batch_count)
                fisher_dict[name] = fisher_dict[name].cpu()

        print(
            f"[EWC] Fisher computed: {sample_count} samples, {batch_count} batches. "
            "Matrices moved to CPU."
        )

        # --- 6. Restore model state ---
        model.train()
        if gc_was_enabled:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            print("[EWC] Gradient checkpointing re-enabled")

        return fisher_dict

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_optimal_params(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """Return a CPU copy of all LoRA parameters (detached)."""
        return {
            name: param.data.clone().detach().cpu()
            for name, param in model.named_parameters()
            if param.requires_grad and "lora" in name.lower()
        }

    def _update_consolidated(
        self,
        new_fisher: Dict[str, torch.Tensor],
        new_params: Dict[str, torch.Tensor],
    ) -> None:
        """
        Merge new Fisher into the consolidated estimate with exponential decay.

        Matches Avalanche's online update:
            F_consolidated = decay * F_old + F_new

        The optimal params are always set to the most recent task's values,
        so the consolidated penalty pulls weights toward the last trained
        checkpoint (weighted by accumulated importance).
        """
        if not self.consolidated_fisher:
            # First experience — just copy
            self.consolidated_fisher = {k: v.clone() for k, v in new_fisher.items()}
        else:
            for name, new_f in new_fisher.items():
                if name in self.consolidated_fisher:
                    old_f = self.consolidated_fisher[name].to(new_f.device)
                    self.consolidated_fisher[name] = (
                        self.decay_factor * old_f + new_f
                    ).cpu()
                else:
                    # New parameter (e.g. new adapter head) — just store it
                    self.consolidated_fisher[name] = new_f.cpu()

        # Always overwrite optimal params with latest task values
        self.consolidated_params = new_params

    def _ewc_loss_online(
        self, model: torch.nn.Module, device: torch.device
    ) -> torch.Tensor:
        if not self.consolidated_fisher:
            return torch.tensor(0.0, device=device)

        ewc_loss = torch.tensor(0.0, device=device)
        for name, param in model.named_parameters():
            if name in self.consolidated_fisher and param.requires_grad:
                fisher = self.consolidated_fisher[name].to(device)
                optimal = self.consolidated_params[name].to(device)
                ewc_loss += (fisher * (param - optimal).pow(2)).sum()

        return self.ewc_lambda * ewc_loss

    def _ewc_loss_separate(
        self, model: torch.nn.Module, device: torch.device
    ) -> torch.Tensor:
        if not self.fisher_dict_list:
            return torch.tensor(0.0, device=device)

        ewc_loss = torch.tensor(0.0, device=device)
        for fisher_dict, optimal_params in zip(
            self.fisher_dict_list, self.optimal_params_list
        ):
            for name, param in model.named_parameters():
                if name in fisher_dict and param.requires_grad:
                    fisher = fisher_dict[name].to(device)
                    optimal = optimal_params[name].to(device)
                    ewc_loss += (fisher * (param - optimal).pow(2)).sum()

        return self.ewc_lambda * ewc_loss

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, save_dir: Path) -> None:
        """Save EWC state to disk."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.mode == "online":
            torch.save(self.consolidated_fisher, save_dir / "consolidated_fisher.pt")
            torch.save(self.consolidated_params, save_dir / "consolidated_params.pt")
        else:
            for idx, (fd, op) in enumerate(
                zip(self.fisher_dict_list, self.optimal_params_list)
            ):
                torch.save(fd, save_dir / f"fisher_exp_{idx}.pt")
                torch.save(op, save_dir / f"optimal_params_exp_{idx}.pt")

        metadata = {
            "ewc_lambda": self.ewc_lambda,
            "mode": self.mode,
            "decay_factor": self.decay_factor,
            "num_experiences": self._num_experiences,
        }
        with open(save_dir / "ewc_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"[EWC] State saved to {save_dir}")

    def load(self, load_dir: Path) -> None:
        """Load EWC state from disk."""
        load_dir = Path(load_dir)

        with open(load_dir / "ewc_metadata.json") as f:
            metadata = json.load(f)

        self.ewc_lambda = metadata["ewc_lambda"]
        self.mode = metadata["mode"]
        self.decay_factor = metadata.get("decay_factor", 0.9)
        self._num_experiences = metadata["num_experiences"]

        if self.mode == "online":
            self.consolidated_fisher = torch.load(
                load_dir / "consolidated_fisher.pt", map_location="cpu"
            )
            self.consolidated_params = torch.load(
                load_dir / "consolidated_params.pt", map_location="cpu"
            )
        else:
            self.fisher_dict_list = []
            self.optimal_params_list = []
            for idx in range(self._num_experiences):
                self.fisher_dict_list.append(
                    torch.load(load_dir / f"fisher_exp_{idx}.pt", map_location="cpu")
                )
                self.optimal_params_list.append(
                    torch.load(
                        load_dir / f"optimal_params_exp_{idx}.pt", map_location="cpu"
                    )
                )

        print(f"[EWC] State loaded from {load_dir} ({self._num_experiences} experiences)")


class RouterEWCRegularizer:
    """
    Elastic Weight Consolidation (EWC) regularizer for CARvE router parameters.

    Applies Fisher-weighted regularization to the router's prompt_projection and
    model_embeddings parameters to prevent catastrophic forgetting of routing
    behaviour learned on previous experiences.

    Unlike EWCRegularizer (which computes Fisher from the LM loss), this class
    receives pre-computed Fisher information from the trainer (which has access to
    the router loss) via add_experience().

    Two modes:
      - 'online': maintains a single consolidated Fisher with exponential decay.
      - 'separate': keeps an independent Fisher snapshot per experience.
    """

    def __init__(
        self,
        ewc_lambda: float = 10000.0,
        mode: str = "online",
        decay_factor: float = 0.9,
    ):
        assert mode in ("separate", "online"), "mode must be 'separate' or 'online'"
        if mode == "online":
            assert 0.0 < decay_factor <= 1.0, "decay_factor must be in (0, 1]"

        self.ewc_lambda = ewc_lambda
        self.mode = mode
        self.decay_factor = decay_factor

        # --- online mode state ---
        self.consolidated_fisher: Dict[str, torch.Tensor] = {}
        self.consolidated_params: Dict[str, torch.Tensor] = {}

        # --- separate mode state ---
        self.fisher_dict_list: List[Dict[str, torch.Tensor]] = []
        self.optimal_params_list: List[Dict[str, torch.Tensor]] = []

        self._num_experiences: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_experience(
        self,
        fisher_dict: Dict[str, torch.Tensor],
        optimal_params: Dict[str, torch.Tensor],
    ) -> None:
        """
        Call this *after* training on each experience with pre-computed Fisher info.

        Args:
            fisher_dict: Mapping of param name → Fisher tensor (CPU). Computed by
                NeighborConsistencySFTTrainer.compute_router_fisher().
            optimal_params: Mapping of param name → parameter value (CPU) taken
                from the trained router immediately after the experience.
        """
        print(f"\n[RouterEWC] Adding experience {self._num_experiences + 1} "
              f"(mode={self.mode}, {len(fisher_dict)} params)...")

        if self.mode == "online":
            self._update_consolidated(fisher_dict, optimal_params)
        else:
            self.fisher_dict_list.append({k: v.clone() for k, v in fisher_dict.items()})
            self.optimal_params_list.append(
                {k: v.clone() for k, v in optimal_params.items()}
            )

        self._num_experiences += 1
        print(f"[RouterEWC] Experience {self._num_experiences} added.")

    def compute_ewc_loss(self, router_model: torch.nn.Module) -> torch.Tensor:
        """
        Compute the EWC regularization penalty for router parameters.

        Loss = ewc_lambda * Σ_i Σ_j  F_ij * (θ_j - θ*_ij)²

        Returns a scalar tensor on the same device as the router model.
        """
        device = next(
            (p.device for p in router_model.parameters() if p.requires_grad),
            torch.device("cpu"),
        )

        if self.mode == "online":
            return self._ewc_loss_online(router_model, device)
        else:
            return self._ewc_loss_separate(router_model, device)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _align_to_stored(
        param: torch.Tensor,
        stored: torch.Tensor,
    ) -> torch.Tensor:
        """Return a slice of `param` that matches the shape of `stored`.

        The router embedding table grows as new models are added across
        experiences.  The stored Fisher / optimal tensors were snapshotted
        when the table was smaller, so we only regularise the rows that
        existed at snapshot time; new rows have zero Fisher and are left
        unconstrained.  For fixed-size parameters (e.g. prompt_projection)
        the shapes match and no slicing is done.
        """
        if param.shape == stored.shape:
            return param
        # Align each dimension independently (handles 1-D and 2-D tensors).
        slices = tuple(slice(0, s) for s in stored.shape)
        return param[slices]

    @staticmethod
    def _common_slices(a: torch.Tensor, b: torch.Tensor) -> tuple:
        """Return per-dimension slices for the overlapping region of a and b."""
        return tuple(slice(0, min(sa, sb)) for sa, sb in zip(a.shape, b.shape))

    def _update_consolidated(
        self,
        new_fisher: Dict[str, torch.Tensor],
        new_params: Dict[str, torch.Tensor],
    ) -> None:
        if not self.consolidated_fisher:
            self.consolidated_fisher = {k: v.clone() for k, v in new_fisher.items()}
        else:
            for name, new_f in new_fisher.items():
                if name in self.consolidated_fisher:
                    old_f = self.consolidated_fisher[name].to(new_f.device)
                    if old_f.shape != new_f.shape:
                        # Robust merge for both growth and shrink:
                        # keep new_f shape, combine only the overlap.
                        merged = new_f.clone()
                        overlap = self._common_slices(old_f, new_f)
                        merged[overlap] = self.decay_factor * old_f[overlap] + new_f[overlap]
                        self.consolidated_fisher[name] = merged.cpu()
                    else:
                        self.consolidated_fisher[name] = (
                            self.decay_factor * old_f + new_f
                        ).cpu()
                else:
                    self.consolidated_fisher[name] = new_f.cpu()

        self.consolidated_params = {k: v.clone() for k, v in new_params.items()}

    def _ewc_loss_online(
        self, router_model: torch.nn.Module, device: torch.device
    ) -> torch.Tensor:
        if not self.consolidated_fisher:
            return torch.tensor(0.0, device=device)

        ewc_loss = torch.tensor(0.0, device=device)
        for name, param in router_model.named_parameters():
            if name in self.consolidated_fisher and param.requires_grad:
                fisher = self.consolidated_fisher[name].to(device)
                optimal = self.consolidated_params[name].to(device)
                # Robust shape handling for both growth and shrink.
                overlap = self._common_slices(param, optimal)
                param_aligned = param[overlap]
                optimal_aligned = optimal[overlap]
                fisher_aligned = fisher[overlap]
                ewc_loss = ewc_loss + (fisher_aligned * (param_aligned - optimal_aligned).pow(2)).sum()

        return self.ewc_lambda * ewc_loss

    def _ewc_loss_separate(
        self, router_model: torch.nn.Module, device: torch.device
    ) -> torch.Tensor:
        if not self.fisher_dict_list:
            return torch.tensor(0.0, device=device)

        ewc_loss = torch.tensor(0.0, device=device)
        for fisher_dict, optimal_params in zip(
            self.fisher_dict_list, self.optimal_params_list
        ):
            for name, param in router_model.named_parameters():
                if name in fisher_dict and param.requires_grad:
                    fisher = fisher_dict[name].to(device)
                    optimal = optimal_params[name].to(device)
                    overlap = self._common_slices(param, optimal)
                    param_aligned = param[overlap]
                    optimal_aligned = optimal[overlap]
                    fisher_aligned = fisher[overlap]
                    ewc_loss = ewc_loss + (fisher_aligned * (param_aligned - optimal_aligned).pow(2)).sum()

        return self.ewc_lambda * ewc_loss

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, save_dir) -> None:
        """Save RouterEWC state to disk."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.mode == "online":
            torch.save(self.consolidated_fisher, save_dir / "router_consolidated_fisher.pt")
            torch.save(self.consolidated_params, save_dir / "router_consolidated_params.pt")
        else:
            for idx, (fd, op) in enumerate(
                zip(self.fisher_dict_list, self.optimal_params_list)
            ):
                torch.save(fd, save_dir / f"router_fisher_exp_{idx}.pt")
                torch.save(op, save_dir / f"router_optimal_params_exp_{idx}.pt")

        metadata = {
            "ewc_lambda": self.ewc_lambda,
            "mode": self.mode,
            "decay_factor": self.decay_factor,
            "num_experiences": self._num_experiences,
        }
        with open(save_dir / "router_ewc_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"[RouterEWC] State saved to {save_dir}")

    def load(self, load_dir) -> None:
        """Load RouterEWC state from disk."""
        load_dir = Path(load_dir)

        with open(load_dir / "router_ewc_metadata.json") as f:
            metadata = json.load(f)

        self.ewc_lambda = metadata["ewc_lambda"]
        self.mode = metadata["mode"]
        self.decay_factor = metadata.get("decay_factor", 0.9)
        self._num_experiences = metadata["num_experiences"]

        if self.mode == "online":
            self.consolidated_fisher = torch.load(
                load_dir / "router_consolidated_fisher.pt", map_location="cpu"
            )
            self.consolidated_params = torch.load(
                load_dir / "router_consolidated_params.pt", map_location="cpu"
            )
        else:
            self.fisher_dict_list = []
            self.optimal_params_list = []
            for idx in range(self._num_experiences):
                self.fisher_dict_list.append(
                    torch.load(
                        load_dir / f"router_fisher_exp_{idx}.pt", map_location="cpu"
                    )
                )
                self.optimal_params_list.append(
                    torch.load(
                        load_dir / f"router_optimal_params_exp_{idx}.pt",
                        map_location="cpu",
                    )
                )

        print(
            f"[RouterEWC] State loaded from {load_dir} "
            f"({self._num_experiences} experiences)"
        )