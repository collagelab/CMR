"""
Custom data collator for router training that preserves metadata fields.

CRITICAL FIX: This collator computes prompt_len by tokenizing the prompt separately,
then uses it to properly mask labels for SFT training.
"""

from typing import Any, Dict, List, Optional
import torch
from transformers import DataCollatorForLanguageModeling

from .router_constants import DEFAULT_ROUTER_EMBEDDING_DIM
from .router_exceptions import RouterTrainingError


class RouterDataCollator:
    """
    Data collator for router training that preserves model_name and domain metadata.
    
    Wraps DataCollatorForLanguageModeling and properly handles prompt/completion masking.
    
    IMPORTANT: Computes prompt_len for each example by tokenizing the prompt separately.
    This ensures labels are correctly masked: -100 for prompt, token_ids for completion.
    """
    
    def __init__(
        self, 
        tokenizer, 
        mlm: bool = False, 
        max_length: int = 1024, 
        **kwargs
    ):
        """
        Initialize the router data collator.
        
        Args:
            tokenizer: The tokenizer to use
            mlm: Whether to use masked language modeling (default: False for causal LM)
            max_length: Maximum sequence length for truncation
            **kwargs: Additional arguments for DataCollatorForLanguageModeling
        """
        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}")
        if tokenizer is None:
            raise ValueError("tokenizer cannot be None")
        
        self.base_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=mlm,
            **kwargs
        )
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Constants for heuristic fallback
        self.HEURISTIC_PROMPT_RATIO = 0.75  # Use 75% of sequence as prompt if no prompt_len available
    
    def _tokenize_and_build_labels(
        self, 
        prompt: str, 
        completion: str
    ) -> Dict[str, Any]:
        """
        Tokenize prompt and completion separately, then build proper input_ids and labels.
        
        This is the CORRECT way to handle SFT training:
        - labels[:prompt_len] = -100 (prompt tokens - no loss)
        - labels[prompt_len:] = token_ids (completion tokens - compute loss)
        - Padding tokens also get -100
        
        Args:
            prompt: The prompt text (up to and including "\n###Response:")
            completion: The completion text (answer + EOS)
        
        Returns:
            Dict with input_ids, attention_mask, labels, prompt_len
        """
        # Tokenize prompt only (without adding EOS)
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        prompt_len = len(prompt_tokens)
        
        # Tokenize completion only (without adding BOS, as it continues from prompt)
        # Remove BOS token if tokenizer adds it
        completion_tokens = self.tokenizer.encode(completion, add_special_tokens=False)
        
        # Combine into full sequence
        input_ids = prompt_tokens + completion_tokens
        
        # Truncate if too long (prefer to keep full prompt, truncate completion)
        if len(input_ids) > self.max_length:
            # Keep full prompt if possible
            if prompt_len <= self.max_length:
                # Truncate completion
                available_for_completion = self.max_length - prompt_len
                input_ids = prompt_tokens + completion_tokens[:available_for_completion]
            else:
                # Prompt itself is too long - truncate it
                # Keep at least 1 token for prompt to avoid zero-length prompts
                prompt_len = self.max_length - 1
                input_ids = prompt_tokens[:prompt_len] + completion_tokens[:1]
        
        # Build labels: -100 for prompt, token_ids for completion
        labels = [-100] * prompt_len + input_ids[prompt_len:]
        
        # Attention mask: 1 for all real tokens
        attention_mask = [1] * len(input_ids)
        
        # After truncation, update prompt_len if it changed
        actual_prompt_len = min(prompt_len, len(input_ids))
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_len": actual_prompt_len,
        }
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate batch and preserve metadata.
        
        Properly handles prompt/completion tokenization and label masking.
        
        Args:
            features: List of examples with prompt/completion strings OR pre-tokenized fields
        
        Returns:
            Batch dict with tensors, metadata, and prompt_len for each example
        """
        # Validate input
        if not features:
            raise RouterTrainingError("Cannot collate empty feature list")
        
        # Extract metadata before collation
        model_names = [f.get("model_name", "unknown") for f in features]
        domains = [f.get("domain", "unknown") for f in features]
        
        # Process features: tokenize if needed, otherwise use pre-tokenized
        processed_features = []
        prompt_lens = []
        
        for idx, f in enumerate(features):
            # Check if already tokenized or needs tokenization
            if "input_ids" in f and "labels" in f:
                # Already tokenized - but check if we need to fix labels
                input_ids = f["input_ids"]
                labels = f["labels"]
                attention_mask = f.get("attention_mask", [1] * len(input_ids))
                
                # Convert to lists if they're tensors (base collator needs lists for padding)
                if isinstance(input_ids, torch.Tensor):
                    input_ids = input_ids.tolist()
                if isinstance(labels, torch.Tensor):
                    labels = labels.tolist()
                if isinstance(attention_mask, torch.Tensor):
                    attention_mask = attention_mask.tolist()
                
                # Ensure they're simple lists (not nested)
                if isinstance(input_ids, list) and len(input_ids) > 0 and isinstance(input_ids[0], list):
                    # Flatten if nested
                    input_ids = input_ids[0] if len(input_ids) == 1 else sum(input_ids, [])
                if isinstance(labels, list) and len(labels) > 0 and isinstance(labels[0], list):
                    labels = labels[0] if len(labels) == 1 else sum(labels, [])
                if isinstance(attention_mask, list) and len(attention_mask) > 0 and isinstance(attention_mask[0], list):
                    attention_mask = attention_mask[0] if len(attention_mask) == 1 else sum(attention_mask, [])
                
                # Check if we have a prompt_len field already
                if "prompt_len" in f:
                    prompt_len = f["prompt_len"]
                elif "prompt" in f and "completion" in f:
                    tokenized = self._tokenize_and_build_labels(f["prompt"], f["completion"])
                    input_ids = tokenized["input_ids"]
                    labels = tokenized["labels"]
                    attention_mask = tokenized["attention_mask"]
                    prompt_len = tokenized["prompt_len"]
                else:
                    prompt_len = max(1, int(len(input_ids) * self.HEURISTIC_PROMPT_RATIO))
                    
                    if not hasattr(self, '_warned_no_prompt_field'):
                        self._warned_no_prompt_field = True
                        # Use warning instead of print for better integration
                        import warnings
                        warnings.warn(
                            f"RouterDataCollator: Features don't have 'prompt'/'completion' fields. "
                            f"Using heuristic: prompt_len = {self.HEURISTIC_PROMPT_RATIO*100}% of sequence length. "
                            f"This may be inaccurate. Consider preserving original text fields.",
                            UserWarning,
                            stacklevel=2
                        )
                
                # Verify and fix labels based on prompt_len
                labels = list(labels)  # Ensure it's a fresh list
                input_ids = list(input_ids)  # Ensure it's a fresh list
                attention_mask = list(attention_mask)  # Ensure it's a fresh list
                
                # Ensure prompt tokens are masked
                for i in range(min(prompt_len, len(labels))):
                    labels[i] = -100
                
                # Ensure completion tokens match input_ids (unless padding)
                for i in range(prompt_len, len(labels)):
                    if i < len(input_ids):
                        labels[i] = input_ids[i]
                
                processed_features.append({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                })
                prompt_lens.append(prompt_len)
                
            elif "prompt" in f and "completion" in f:
                # Need to tokenize
                tokenized = self._tokenize_and_build_labels(f["prompt"], f["completion"])
                
                processed_features.append({
                    "input_ids": tokenized["input_ids"],
                    "attention_mask": tokenized["attention_mask"],
                    "labels": tokenized["labels"],
                })
                prompt_lens.append(tokenized["prompt_len"])
            else:
                raise RouterTrainingError(
                    f"Feature {idx} must have either (prompt, completion) or (input_ids, labels). "
                    f"Got keys: {list(f.keys())}. "
                    f"This indicates a data format issue - each feature must be tokenized or have raw text."
                )
        
        # Pad manually instead of using base collator (which has issues with our data structure)
        # Find max length in batch
        max_len = max(len(f["input_ids"]) for f in processed_features)
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        
        # Pad each feature
        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []
        
        for f in processed_features:
            seq_len = len(f["input_ids"])
            padding_len = max_len - seq_len
            
            # Pad input_ids
            padded_input_ids.append(f["input_ids"] + [pad_token_id] * padding_len)
            
            # Pad attention_mask (0 for padding)
            padded_attention_mask.append(f["attention_mask"] + [0] * padding_len)
            
            # Pad labels (-100 for padding)
            padded_labels.append(f["labels"] + [-100] * padding_len)
        
        # Convert to tensors
        batch = {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }
        
        # Add prompt_lens to batch (pad to batch if needed due to collation)
        # The collator may have added padding, so we need to keep prompt_lens aligned
        batch["prompt_len"] = torch.tensor(prompt_lens, dtype=torch.long)
        
        # Add metadata back (keep as lists, not tensors)
        batch["model_name"] = model_names
        batch["domain"] = domains
        
        return batch
    
