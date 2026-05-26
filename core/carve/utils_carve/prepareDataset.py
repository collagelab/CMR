from typing import Dict, List, Optional
import json
import random
from collections import defaultdict
import torch
from .configs import TrainConfig, ApibenchDataConfig, MLLMDataConfig
from datasets import Dataset
from .retrieval_replay import PromptReplayBuffer, ExperienceIndex, generate_example_id

dict_retriever = {
    "bm25": "bm25_retrieved_info",
    "sentence_transformer": "sentence_transformer_retrieved_info",
    "splade": "splade_retrieved_info", 
    "flagembedding": "flagembedding_retrieved_info",
}

gorilla_prompt = (
    "You are Gorilla, an expert API model router. "
    "Read the ###Instruction and ###Input below and return ONLY a single model name. "
    "Do not invent model name. Do not return anything else.\n\n"
)

gorilla_fewshot_prompt = (
    "You are Gorilla, an expert API model router. "
    "Read the [ORIGINAL PROMPT] section below and return ONLY a single model name for that prompt. "
    "If [RELATED EXAMPLES] are provided, they are for reference only to help you understand similar cases - "
    "do NOT return the models from those examples. Return a model name only for the [ORIGINAL PROMPT]. "
    "Do not invent model name. Do not return anything else.\n\n"
)

def create_gorilla_prompt_with_date(date: str) -> str:
    """Create a gorilla prompt with a date cutoff for model selection."""
    return (
        "You are Gorilla, an expert API model router. "
        "Read the ###Instruction and ###Input below and return ONLY a single model name. "
        f"Do not invent model name. Do not return anything else. Choose only models with a model date before {date}.\n\n"
    )

def create_gorilla_fewshot_prompt_with_date(date: str) -> str:
    """Create a gorilla few-shot prompt with a date cutoff for model selection."""
    return (
        "You are Gorilla, an expert API model router. "
        "Read the [ORIGINAL PROMPT] section below and return ONLY a single model name for that prompt. "
        "If [RELATED EXAMPLES] are provided, they are for reference only to help you understand similar cases - "
        "do NOT return the models from those examples. Return a model name only for the [ORIGINAL PROMPT]. "
        f"Choose only models with a model date before {date}. "
        "Do not invent model name. Do not return anything else.\n\n"
    )

gorilla_prompt_explanation_json = (
    "You are Gorilla, an expert API model router. "
    "Read the ###Instruction and ###Input below and answer with a model name and a brief explanation. "
    "Return the answer in JSON format with fields 'model_name' and 'explanation'. "
    "For example: {\"model_name\": \"actual model name\", \"explanation\": \"actual explanation\"}. "
    "Do not include any other fields. Do not invent model names. Do not return anything else. "
    "Important: DO NOT wrap the JSON in triple backticks or any markdown/code fence (for example, do NOT return ```json ... ```). "
    "Return only the JSON object string starting with '{' and ending with '}', with no surrounding markdown, backticks, or extra text. "
    "Do not prepend or append any characters outside the JSON (no headings, no explanatory text).\n\n"
)

gorilla_fewshot_prompt_explanation_json = (
    "You are Gorilla, an expert API model router. "
    "Read the [ORIGINAL PROMPT] section below and answer with a model name and a brief explanation for that prompt. "
    "If [RELATED EXAMPLES] are provided, they are for reference only to help you understand similar cases - "
    "do NOT return the models from those examples. Return a model name only for the [ORIGINAL PROMPT]. "
    "Return the answer in JSON format with fields 'model_name' and 'explanation'. "
    "For example: {\"model_name\": \"actual model name\", \"explanation\": \"actual explanation\"}. "
    "Do not include any other fields. Do not invent model names. Do not return anything else. "
    "Important: DO NOT wrap the JSON in triple backticks or any markdown/code fence (for example, do NOT return ```json ... ```). "
    "Return only the JSON object string starting with '{' and ending with '}', with no surrounding markdown, backticks, or extra text. "
    "Do not prepend or append any characters outside the JSON (no headings, no explanatory text).\n\n"
)

def create_gorilla_prompt_explanation_json_with_date(date: str) -> str:
    """Create a gorilla prompt with explanation (JSON format) and a date cutoff for model selection."""
    return (
        "You are Gorilla, an expert API model router. "
        "Read the ###Instruction and ###Input below and answer with a model name and a brief explanation. "
        "Return the answer in JSON format with fields 'model_name' and 'explanation'. "
        "For example: {\"model_name\": \"actual model name\", \"explanation\": \"actual explanation\"}. "
        "Do not include any other fields. Do not invent model names. Do not return anything else. "
        f"Choose only models with a model date before {date}. "
        "Important: DO NOT wrap the JSON in triple backticks or any markdown/code fence (for example, do NOT return ```json ... ```). "
        "Return only the JSON object string starting with '{' and ending with '}', with no surrounding markdown, backticks, or extra text. "
        "Do not prepend or append any characters outside the JSON (no headings, no explanatory text).\n\n"
    )

def create_gorilla_fewshot_prompt_explanation_json_with_date(date: str) -> str:
    """Create a gorilla few-shot prompt with explanation (JSON format) and a date cutoff for model selection."""
    return (
        "You are Gorilla, an expert API model router. "
        "Read the [ORIGINAL PROMPT] section below and answer with a model name and a brief explanation for that prompt. "
        "If [RELATED EXAMPLES] are provided, they are for reference only to help you understand similar cases - "
        "do NOT return the models from those examples. Return a model name only for the [ORIGINAL PROMPT]. "
        "Return the answer in JSON format with fields 'model_name' and 'explanation'. "
        "For example: {\"model_name\": \"actual model name\", \"explanation\": \"actual explanation\"}. "
        "Do not include any other fields. Do not invent model names. Do not return anything else. "
        f"Choose only models with a model date before {date}. "
        "Important: DO NOT wrap the JSON in triple backticks or any markdown/code fence (for example, do NOT return ```json ... ```). "
        "Return only the JSON object string starting with '{' and ending with '}', with no surrounding markdown, backticks, or extra text. "
        "Do not prepend or append any characters outside the JSON (no headings, no explanatory text).\n\n"
    )

gorilla_prompt_explanation = (
    "You are Gorilla, an expert API model router. "
    "Read the ###Instruction and ###Input below and answer with a model name and a brief explanation. "
    "Return the answer in the format: <<<model_name>>>:actual model name <<<explanation>>>:actual explanation. "
    "Do not include any other fields. Do not invent model names. Do not return anything else.\n\n"
)

gorilla_fewshot_prompt_explanation = (
    "You are Gorilla, an expert API model router. "
    "Read the [ORIGINAL PROMPT] section below and answer with a model name and a brief explanation for that prompt. "
    "If [RELATED EXAMPLES] are provided, they are for reference only to help you understand similar cases - "
    "do NOT return the models from those examples. Return a model name only for the [ORIGINAL PROMPT]. "
    "Return the answer in the format: <<<model_name>>>:actual model name <<<explanation>>>:actual explanation. "
    "Do not include any other fields. Do not invent model names. Do not return anything else.\n\n"
)

def create_gorilla_prompt_explanation_with_date(date: str) -> str:
    """Create a gorilla prompt with explanation (gorilla format) and a date cutoff for model selection."""
    return (
        "You are Gorilla, an expert API model router. "
        "Read the ###Instruction and ###Input below and answer with a model name and a brief explanation. "
        "Return the answer in the format: <<<model_name>>>:actual model name <<<explanation>>>:actual explanation. "
        "Do not include any other fields. Do not invent model names. Do not return anything else. "
        f"Choose only models with a model date before {date}.\n\n"
    )

def create_gorilla_fewshot_prompt_explanation_with_date(date: str) -> str:
    """Create a gorilla few-shot prompt with explanation (gorilla format) and a date cutoff for model selection."""
    return (
        "You are Gorilla, an expert API model router. "
        "Read the [ORIGINAL PROMPT] section below and answer with a model name and a brief explanation for that prompt. "
        "If [RELATED EXAMPLES] are provided, they are for reference only to help you understand similar cases - "
        "do NOT return the models from those examples. Return a model name only for the [ORIGINAL PROMPT]. "
        "Return the answer in the format: <<<model_name>>>:actual model name <<<explanation>>>:actual explanation. "
        "Do not include any other fields. Do not invent model names. Do not return anything else. "
        f"Choose only models with a model date before {date}.\n\n"
    )

# string template for answers in json format
def create_json_answer_template(model_name: str, explanation: str) -> str:
    return json.dumps({
        "model_name": model_name,
        "explanation": explanation
    })

# string template for answers in gorilla format with explanation
def create_gorilla_explanation_answer_template(model_name: str, explanation: str) -> str:
    return f"<<<model_name>>>:{model_name} <<<explanation>>>:{explanation}"
    

def apply_label_noise(
    raw_data: List[Dict[str, str]],
    noise_prob: float,
    noise_target: str = "model",
    noise_mode: str = "random",
    seed: Optional[int] = None,
) -> List[Dict[str, str]]:
    """
    Randomly corrupt model_name and/or domain labels in raw training examples.

    Operates on raw data before conversion so both the completion text and
    router metadata columns are corrupted consistently.

    Args:
        raw_data:     List of raw data entries (each a dict with model_name, domain, etc.)
        noise_prob:   Probability [0, 1] of corrupting each example.
        noise_target: Which label(s) to corrupt: "model", "domain", or "both".
        noise_mode:   How to sample the replacement:
                        "random"      – uniform draw from the whole pool.
                        "same_domain" – draw from the same domain as the entry
                                        (harder noise for model, no-op for domain).
        seed:         Optional RNG seed for reproducibility.

    Returns:
        A new list with (possibly) corrupted entries; original dicts are not mutated.
    """
    if noise_prob <= 0.0:
        return raw_data

    rng = random.Random(seed)

    # Build pools from the full dataset
    all_models = list({e.get("model_name", "") for e in raw_data if e.get("model_name")})
    all_domains = list({e.get("domain", "") for e in raw_data if e.get("domain")})

    domain_to_models: Dict[str, List[str]] = defaultdict(list)
    if noise_mode == "same_domain":
        for e in raw_data:
            m, d = e.get("model_name", ""), e.get("domain", "")
            if m and d:
                domain_to_models[d].append(m)
        domain_to_models = {d: list(set(ms)) for d, ms in domain_to_models.items()}

    corrupt_model = noise_target in ("model", "both")
    corrupt_domain = noise_target in ("domain", "both")

    noisy: List[Dict[str, str]] = []
    n_corrupted = 0
    for entry in raw_data:
        if rng.random() >= noise_prob:
            noisy.append(entry)
            continue

        entry = dict(entry)  # shallow copy – don't mutate the original

        if corrupt_model:
            current_model = entry.get("model_name", "")
            if noise_mode == "same_domain":
                domain = entry.get("domain", "")
                candidates = [m for m in domain_to_models.get(domain, []) if m != current_model]
                if not candidates:
                    candidates = [m for m in all_models if m != current_model]
            else:
                candidates = [m for m in all_models if m != current_model]
            if candidates:
                entry["model_name"] = rng.choice(candidates)

        if corrupt_domain:
            current_domain = entry.get("domain", "")
            candidates = [d for d in all_domains if d != current_domain]
            if candidates:
                entry["domain"] = rng.choice(candidates)

        n_corrupted += 1
        noisy.append(entry)

    print(
        f"[LabelNoise] Corrupted {n_corrupted}/{len(raw_data)} examples "
        f"(target={noise_target}, mode={noise_mode}, prob={noise_prob})"
    )
    return noisy


def load_dataset_json(path: str) -> list:
    dataset_json = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dataset_json.append(json.loads(line))
            else:
                raise ValueError("Empty line found in the dataset file.")
    return dataset_json

def convert_to_conversational(
    raw_data: List[Dict[str, str]], 
    config: TrainConfig, 
    tokenizer,
    dataset_config: Optional[object] = None,
    is_replay: bool = False,
) -> List[Dict[str, str]]:
    # Keep metadata columns only when losses/samplers need them.
    use_metadata_columns = (
        (hasattr(config, "loss_mode") and config.loss_mode in ["router", "router+graph", "supervised+router", "supervised+router+graph"])
    )
    # Use custom system prompt if provided, otherwise use default prompts.
    if config.system_prompt != "":
        system_prompt = config.system_prompt
    else:
        # Determine which prompt template to use based on system_prompt_format.
        # For standard supervised conversion, use non-fewshot gorilla prompts.
        if config.system_prompt_format == "gorilla_prompt_explanation_json":
            system_prompt = gorilla_prompt_explanation_json
        elif config.system_prompt_format == "gorilla_prompt_explanation":
            system_prompt = gorilla_prompt_explanation
        else:
            # Default to gorilla_prompt
            system_prompt = gorilla_prompt
   
    # Apply label noise when configured (skipped for replay examples unless label_noise_replay=True)
    noise_prob = getattr(config, "label_noise_prob", 0.0)
    if noise_prob > 0.0 and (not is_replay or getattr(config, "label_noise_replay", False)):
        raw_data = apply_label_noise(
            raw_data=raw_data,
            noise_prob=noise_prob,
            noise_target=getattr(config, "label_noise_target", "model"),
            noise_mode=getattr(config, "label_noise_mode", "random"),
            seed=getattr(config, "seed", None),
        )

    conversational_dataset = []
    for entry in raw_data:
        prompt = entry.get("instruction", "").replace('\r\n', '\n').strip()
        model_name = entry.get("model_name", "").replace('\r\n', '\n').strip()
        
        explanation = entry.get("explanation", "").replace('\r\n', '\n').strip()
        if len(explanation) > 1000:
            explanation = explanation[:1000] + "..."  # truncate long explanations, only few of them exceed 1000 chars, usually due to bad data entries, truncate to avoid very long inputs and save memory
            
        if config.system_prompt_format == "gorilla_prompt_explanation_json":
            answer = create_json_answer_template(model_name, explanation)
        elif config.system_prompt_format == "gorilla_prompt_explanation":
            answer = create_gorilla_explanation_answer_template(model_name, explanation)
        else:
            answer = model_name

        # Retrieve model_card
        model_card = ""
        if config.retriever:
            try:
                retriever_name = dict_retriever[config.retriever]
                retrieved_info = entry.get(retriever_name, "")
                if retrieved_info:
                    model_card = "\n<Reference API>: " + retrieved_info.replace('\r\n', '\n').strip()
            except KeyError:
                print(
                    f"Retriever '{config.retriever}' is not valid. Choose from: {list(dict_retriever.keys())}")

        if prompt and model_name:
            # Build the full prompt without extra stripping
            full_prompt = system_prompt + prompt + model_card + "\n###Response:"
            
            row = {
                "prompt": full_prompt,
                "completion": " " + answer + tokenizer.eos_token,  # Keep consistent leading space
            }
            if use_metadata_columns:
                row["is_replay"] = is_replay  # Flag for replay-aware losses
                row["model_name"] = model_name  # For contrastive/router bookkeeping
                row["domain"] = entry.get("domain", "")  # Domain for filtering
            conversational_dataset.append(row)
        else:
            raise ValueError("Both 'instruction' and 'model_name' must be present in each data entry.")

    dataset = Dataset.from_list(conversational_dataset)
    return dataset


def truncate_text_by_tokens(text: str, max_tokens: int) -> str:
    """
    Truncate text to approximately max_tokens by splitting on spaces.
    Simple approximation: 1 token ≈ 4 characters.
    """
    if not text:
        return ""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    # Truncate and add ellipsis
    truncated = text[:max_chars].rsplit(' ', 1)[0]  # Cut at word boundary
    return truncated + "..."


def convert_to_retrieval_replay_fewshot(
    raw_data: List[Dict[str, str]], 
    config: TrainConfig, 
    tokenizer,
    dataset_config: Optional[object] = None,
    experience_idx: int = 0,
    experience_name: str = "",
    replay_buffer: Optional[PromptReplayBuffer] = None,
    experience_index: Optional[ExperienceIndex] = None
) -> Dataset:
    """
    Convert dataset to conversational format with few-shot augmentation using retrieval replay.
    
    Args:
        raw_data: List of raw data entries
        config: TrainConfig with few-shot settings
        tokenizer: Tokenizer for formatting
        dataset_config: Optional dataset config
        experience_idx: Index of current experience (0-based)
        experience_name: Name of current experience
        replay_buffer: Optional replay buffer with previous experience examples
        experience_index: Optional pre-built experience index (if None, will build one)
    """
    if not config.retriever:
        raise ValueError("retriever must be specified for retrieval_replay_fewshot baseline")
    
    # Use custom system prompt if provided, otherwise use default prompts.
    if config.system_prompt != "":
        system_prompt = config.system_prompt
    else:
        # Determine which prompt template to use based on system_prompt_format
        # Use few-shot variants for retrieval_replay_fewshot
        if config.system_prompt_format == "gorilla_prompt_explanation_json":
            system_prompt = gorilla_fewshot_prompt_explanation_json
        elif config.system_prompt_format == "gorilla_prompt_explanation":
            system_prompt = gorilla_fewshot_prompt_explanation
        else:
            # Default to gorilla_fewshot_prompt
            system_prompt = gorilla_fewshot_prompt
    
    # Build experience index if not provided
    if experience_index is None:
        replay_examples = replay_buffer.get_examples() if replay_buffer else []
        # Determine device for retriever
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        experience_index = ExperienceIndex(
            retriever_type=config.retriever,
            current_examples=raw_data,
            replay_examples=replay_examples,
            experience_name=experience_name,
            device=device
        )
    
    conversational_dataset = []
    top_k = config.fewshot_top_k
    max_card_tokens = config.fewshot_max_card_tokens
    dropout_prob = config.fewshot_dropout_prob
    
    # Set random seed for reproducibility if provided
    if config.seed is not None:
        random.seed(config.seed)
    
    # Debug: print a few examples
    num_examples_to_print = 3
    examples_printed = 0
    
    for local_idx, entry in enumerate(raw_data):
        prompt = entry.get("instruction", "").replace('\r\n', '\n').strip()
        model_name = entry.get("model_name", "").replace('\r\n', '\n').strip()
        
        if not prompt or not model_name:
            continue
        
        # Generate example_id for self-masking
        example_id = entry.get("example_id")
        if not example_id:
            example_id = generate_example_id(prompt, model_name, experience_name, local_idx)
        
        # Decide whether to use few-shot examples based on dropout probability
        use_fewshot = random.random() > dropout_prob
        
        # Build few-shot examples section (only if not dropping out)
        fewshot_section = ""
        retrieved_neighbors = None
        if use_fewshot:
            # Retrieve similar prompts with self-masking
            retrieved_neighbors = experience_index.retrieve(
                query=prompt,
                top_k=top_k,
                exclude_example_ids=[example_id]
            )
            
            if retrieved_neighbors:
                fewshot_section = "\n\n[RELATED EXAMPLES]\n"
                for i, neighbor in enumerate(retrieved_neighbors, 1):
                    neighbor_prompt = neighbor['prompt']
                    neighbor_model = neighbor['model_id']
                    neighbor_domain = neighbor.get('domain', '')
                    neighbor_card = truncate_text_by_tokens(
                        neighbor.get('model_card_snippet', ''), 
                        max_card_tokens
                    )
                    
                    fewshot_section += f"Example {i}:\n"
                    fewshot_section += f"  Prompt: {neighbor_prompt}\n"
                    fewshot_section += f"  Reference model (for similar case): {neighbor_model}\n"
                    if neighbor_domain:
                        fewshot_section += f"  Domain: {neighbor_domain}\n"
                    if neighbor_card:
                        fewshot_section += f"  Model card: {neighbor_card}\n"
                    fewshot_section += "\n"
        
        # Build augmented prompt
        augmented_prompt = f"[ORIGINAL PROMPT]\n{prompt}"
        if fewshot_section:
            augmented_prompt += fewshot_section
        
        # Get answer for dataset
        explanation = entry.get("explanation", "").replace('\r\n', '\n').strip()
        if len(explanation) > 1000:
            explanation = explanation[:1000] + "..."
        
        if config.system_prompt_format == "gorilla_prompt_explanation_json":
            answer = create_json_answer_template(model_name, explanation)
        elif config.system_prompt_format == "gorilla_prompt_explanation":
            answer = create_gorilla_explanation_answer_template(model_name, explanation)
        else:
            answer = model_name
        
        # Debug: Print examples of retrieved neighbors and final prompts
        if examples_printed < num_examples_to_print:
            print(f"\n{'='*80}")
            print(f"EXAMPLE {examples_printed + 1} - Training Example Index: {local_idx}")
            print(f"{'='*80}")
            print(f"\n[ORIGINAL PROMPT]:")
            print(f"{prompt[:200]}{'...' if len(prompt) > 200 else ''}")
            print(f"\n[EXPECTED MODEL]: {model_name}")
            print(f"[DOMAIN]: {entry.get('domain', 'N/A')}")
            
            if use_fewshot and retrieved_neighbors:
                print(f"\n[RETRIEVED SIMILAR PROMPTS] ({len(retrieved_neighbors)} examples):")
                for i, neighbor in enumerate(retrieved_neighbors, 1):
                    print(f"\n  Neighbor {i}:")
                    print(f"    Prompt: {neighbor['prompt'][:150]}{'...' if len(neighbor['prompt']) > 150 else ''}")
                    print(f"    Selected Model: {neighbor['model_id']}")
                    print(f"    Domain: {neighbor.get('domain', 'N/A')}")
                    if neighbor.get('model_card_snippet'):
                        card_preview = neighbor['model_card_snippet'][:100]
                        print(f"    Model Card: {card_preview}{'...' if len(neighbor['model_card_snippet']) > 100 else ''}")
            elif use_fewshot:
                print(f"\n[RETRIEVED SIMILAR PROMPTS]: None found")
            else:
                print(f"\n[FEW-SHOT DROPOUT]: Using original prompt without examples (dropout_prob={dropout_prob})")
            
            print(f"\n[FULL TRAINING PROMPT] (first 800 chars):")
            full_prompt_preview = (system_prompt + augmented_prompt + "\n###Response:")[:800]
            print(f"{full_prompt_preview}...")
            print(f"\n[EXPECTED COMPLETION]: {answer[:100]}{'...' if len(answer) > 100 else ''}")
            print(f"{'='*80}\n")
            examples_printed += 1
        
        full_prompt = system_prompt + augmented_prompt + "\n###Response:"
        
        conversational_dataset.append({
            "prompt": full_prompt,
            "completion": " " + answer + tokenizer.eos_token
        })
    
    dataset = Dataset.from_list(conversational_dataset)
    return dataset

