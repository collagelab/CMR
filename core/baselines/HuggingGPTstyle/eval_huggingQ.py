import argparse
import json
import re
import sys
from pathlib import Path

def _find_repo_root(start_path: Path) -> Path:
    for parent in [start_path] + list(start_path.parents):
        if (parent / "pyproject.toml").exists() or (parent / "core").is_dir():
            return parent
    return start_path


# Ensure repo root is on sys.path so local imports work from any CWD.
PROJECT_ROOT = _find_repo_root(Path(__file__).resolve())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import Levenshtein  

def load_dataset_json(path: str) -> list:
    dataset_json = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dataset_json.append(json.loads(line))
            else:
                raise ValueError("Empty line found in the dataset file.")
    return dataset_json

def levenshtein_similarity(name1, name2):
    max_len = max(len(name1), len(name2))
    if max_len == 0:
        return 1.0  # If both strings are empty, consider them identical
    return 1 - (Levenshtein.distance(name1, name2) / max_len)

def normalize_string(s):
    """
    Normalize a string for comparison:
    - Convert to lowercase
    - Strip leading/trailing whitespace
    - Remove extra whitespace (replace multiple spaces with single space)
    - Remove punctuation at the end
    """
    if not s:
        return ""

    # Convert to lowercase and strip
    s = s.lower().strip()

    # Replace multiple whitespaces with single space
    s = re.sub(r"\s+", " ", s)

    # Remove trailing punctuation
    s = re.sub(r"[.,;:!?]+$", "", s)

    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input JSON file containing model responses.",
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        required=True,
        help="Path to the dataset JSON file containing ground truth.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=False,
        help="Path to the output JSON file to save evaluation results.",
    )
    args = parser.parse_args()

    # Load input data (model results)
    with open(args.input_file, "r") as f:
        data_qwen = json.load(f)
    print(f"Loaded {len(data_qwen)} entries from {args.input_file}")

    # Load dataset (ground truth)
    dataset_true = load_dataset_json(args.dataset_file)
    print(f"Loaded {len(dataset_true)} entriesb from {args.dataset_file}")

    # Extract domains from data_qwen['generated_text']
    pattern = r"If uncertain, choose the closest matching domain.\n\n(.*?)###Response"

    for entry in data_qwen:
        # Handle generated_text: take first element if it's a list
        gen = entry.get("generated_text", "")
        entry["domain"] = gen[0] if isinstance(gen, list) and gen else gen

        # Extract instruction using a single pattern
        prompt = entry.get("prompt", "")
        match = re.search(pattern, prompt, re.DOTALL)
        entry["instruction"] = match.group(1).strip() if match else ""
    ################################################################################
    correct = 0
    for entry_qwen, entry_true in zip(data_qwen, dataset_true):
        domain_qwen = normalize_string(entry_qwen.get("domain", ""))
        domain_true = normalize_string(entry_true.get("domain", ""))
        if domain_qwen == domain_true:
            correct += 1
        # check same prompt
        prompt_qwen = normalize_string(entry_qwen.get("instruction", ""))
        # if entry_true['instruction'] not start with ###Instruction: add it
        instruction_true = entry_true.get("instruction", "")
        if not instruction_true.startswith("###Instruction:"):
            instruction_true = "###Instruction: " + instruction_true
        prompt_true = normalize_string(instruction_true)
        if prompt_qwen != prompt_true:
            print(
                f"⚠️ Warning: Mismatched prompts:\nQwen: {prompt_qwen}\nTrue: {prompt_true}\n"
            )

    accuracy = correct / len(data_qwen) if len(data_qwen) > 0 else 0.0
    print(f"\n✅ Accuracy: {accuracy:.4f} ({correct}/{len(data_qwen)})")


if __name__ == "__main__":
    main()
