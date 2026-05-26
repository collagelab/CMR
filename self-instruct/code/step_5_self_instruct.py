import argparse
import json
import random
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import os

# =========================
# PROMPT CONSTRUCTION
# =========================

SYSTEM_PROMPT = """You are an expert in natural language query generation and API utilization. 
Your task is to generate 20 diverse and creative user queries that can interact with a given API function, based on its description, while strictly following the rules below:

Rules:
1. Output must be a single JSON object with the key "queries" containing an array of 20 query strings.
2. Each query should be natural, human-like, and clearly indicate a use of the API’s functionality.
3. Do NOT include the API's name or any of the prohibited words in the queries.
4. Queries must be diverse in context, style, and phrasing.
5. Ensure each query is self-contained and understandable without external context.
6. Avoid repetition, generic queries, or references to programming concepts unless the API explicitly involves code.

Input Format:
{
  "API Name": "[API Name]",
  "Description": "[Detailed API description]",
  "Prohibit Words": ["API", "model", "tool", "tools", "use", "function", "method"]
}

Output Format:
{
  "queries": [
    "Query 1: ...",
    "Query 2: ...",
    "Query 3: ...",
    "Query 4: ...",
    "Query 5: ...",
    "Query 6: ...",
    "Query 7: ...",
    "Query 8: ...",
    "Query 9: ...",
    "Query 10: ...",
    "Query 11: ...",
    "Query 12: ...",
    "Query 13: ...",
    "Query 14: ...",
    "Query 15: ...",
    "Query 16: ...",
    "Query 17: ...",
    "Query 18: ...",
    "Query 19: ...",
    "Query 20: ..."
  ]
}

Instructions:
- Replace the "..." in each query with a complete, natural, human-like instruction utilizing the API's capabilities.
- Ensure all queries adhere to the rules above.
- The JSON output must be valid and parsable.

Now, generate the JSON output for the following input:

Input:
{
  "API Name": "{model_id}",
  "Description": "{modelcard}",
  "Prohibit Words": ["API", "model", "tool", "tools", "use", "function", "method", "{model_id}"]
}

"""


def build_prompt(model_info: dict) -> str:
    prompt = (
        SYSTEM_PROMPT.replace("{model_id}", model_info["model_id"])
        .replace("{modelcard}", model_info["modelcard"])
        .replace("{model_id}", model_info["model_id"])
    )
    return prompt


def convert_to_chat(prompts: list, tokenizer, enable_thinking=False) -> str:
    chat_texts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        chat_texts.append(text)
    return chat_texts


# =========================
# MAIN
# =========================


def main():
    parser = argparse.ArgumentParser(
        description="Self-instruct: create prompts from model cards using a LLM."
    )
    parser.add_argument(
        "--model_cards_path",
        type=str,
        required=True,
        help="path of the model cards file (.jsonl)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save JSON outputs",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="path to LLM model",
    )
    args = parser.parse_args()

    model_cards_path = Path(args.model_cards_path)

    all_models = []

    with open(model_cards_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                all_models.append(json.loads(line))
    # shuffle all_models for diversity

    # random.shuffle(all_models)

    # =========================
    # LOAD MODEL
    # =========================
    MAX_OUTPUT_LEN = 15000  # max_model_len of your LLM
    MAX_MODEL_LEN = 40960  # 40960 is the standard

    llm = LLM(
        model=args.model_path,
        dtype="bfloat16",
        tensor_parallel_size=1,
        # gpu_memory_utilization=0.7,
    )
    # adjust sampling params as needed
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.9,
        max_tokens=MAX_OUTPUT_LEN,
    )

    # =========================
    # LOAD JSONL + BUILD PROMPTS
    # =========================

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    print(f"ℹ️  Loaded {len(all_models)} model cards.")

    # Filter models within token limit and build prompts
    collected_models = []
    safe_prompts = []

    for entry in all_models:
        # api info is a dict of model_id modelcard domain
        api_info = {
            "model_id": entry["model_id"],
            "modelcard": entry["modelcard"],
            "domain": entry["domain"],
        }
        prompt_text = build_prompt(api_info)
        num_tokens = tokenizer(
            prompt_text, return_tensors="pt", truncation=False
        ).input_ids.shape[1]

        if num_tokens >= MAX_MODEL_LEN:
            print(
                f"⚠️ Skipping {entry.get('model_id', 'unknown')} ({num_tokens} tokens)"
            )
            continue

        # Only add to both lists if within token limit
        collected_models.append(entry)
        safe_prompts.append(prompt_text)

    print(
        f"✅ {len(collected_models)} model cards within token limit (same as {len(safe_prompts)} prompts)."
    )

    # =========================
    # SINGLE BATCH INFERENCE
    # =========================
    # collected_models = collected_models[:10]
    # safe_prompts = safe_prompts[:10]
    chat_prompts = convert_to_chat(safe_prompts, tokenizer, enable_thinking=False)
    outputs = llm.generate(chat_prompts, sampling_params)

    # =========================
    # POST-PROCESSING
    # =========================

    for entry, output in zip(collected_models, outputs):
        # model_id = entry["model_id"]
        generated = output.outputs[0].text.strip()
        entry["generated"] = generated

    # =========================
    # WRITE SINGLE OUTPUT FILE
    # =========================

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(collected_models, f, ensure_ascii=False, indent=2)

    print(f"\nCompleted. Processed {len(collected_models)} model cards.")


if __name__ == "__main__":
    main()
