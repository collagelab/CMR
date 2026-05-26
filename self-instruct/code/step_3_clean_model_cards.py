import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

import os


# =========================
# PROMPT CONSTRUCTION
# =========================


SYSTEM_PROMPT = """You are a model card normalization system.

Your task is to extract and REWRITE the text that DESCRIBES the MODEL,
producing a clean, continuous, neutral description.

The goal is NOT to summarize, but to REMOVE NOISE while preserving
ALL factual information and level of detail.

────────────────────────────────────────
OBJECTIVE
────────────────────────────────────────
Produce a readable description of the model that explains:
- what the model is
- what it is designed to do
- what tasks or functions it supports, as stated
- its architecture or design, if mentioned
- its performance or evaluation results, if mentioned
- any explicit caveats or warnings, in context

The output MUST remain semantically equivalent to the original description
and MUST NOT reduce informational content.

────────────────────────────────────────
CONTENT SELECTION RULES
────────────────────────────────────────
Include:
- All descriptive and explanatory statements about the model
- All stated tasks or capabilities expressed in sentences
- All qualitative or quantitative performance claims
- All explicit caveats, warnings, or limitations

Exclude COMPLETELY:
- Tables of any kind
- Raw benchmark or leaderboard dumps
- Output formats, schemas, or example JSON
- Code, pseudocode, or API instructions
- UI or navigation text
- Symbols, markdown artifacts, or formatting noise
- Headings or task names without descriptive sentences

If information appears only inside tables, you MAY restate it
in sentence form, preserving the same facts and numbers.

────────────────────────────────────────
REWRITING RULES (STRICT)
────────────────────────────────────────
- You MAY paraphrase sentences.
- You MUST preserve ALL facts, numbers, and qualifiers.
- You MUST NOT omit, abstract, or compress information.
- You MUST NOT introduce structure not present in the original text.
- You MUST NOT add interpretations or inferences.

────────────────────────────────────────
FAILURE CONDITION (MANDATORY)
────────────────────────────────────────
If the model card contains NO text that describes the model under the rules above,
return EXACTLY the string:

None

────────────────────────────────────────
OUTPUT CONSTRAINTS (MANDATORY)
────────────────────────────────────────
- Output MUST be valid JSON.
- Output MUST contain exactly one field: "model_description".
- The value MUST be either:
  - a plain text string, or
  - the exact string "None".
- Do NOT use markdown, lists, tables, or symbols.

────────────────────────────────────────
OUTPUT FORMAT (EXACT)
────────────────────────────────────────
{
  "model_description": "<clean description or None>"
}

MODEL CARD TEXT:
{model_card_text}
"""


def build_prompt(model_card_text: str) -> str:
    prompt = SYSTEM_PROMPT.replace("{model_card_text}", model_card_text.strip())
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
        description="Extract model descriptions from AI model cards using a LLM."
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
        help="Path to save extracted JSON outputs",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="path to LLM model",
    )
    args = parser.parse_args()

    model_cards_path = Path(args.model_cards_path)
    # output_dir = Path(args.output_dir)
    # output_dir.mkdir(parents=True, exist_ok=True)

    all_models = []

    with open(model_cards_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            all_models.append(entry)

    # =========================
    # LOAD MODEL
    # =========================
    MAX_OUTPUT_LEN = 15000  # max_model_len of your LLM
    MAX_MODEL_LEN = 40960  # 40960 is the standard for Qwen3-32B

    llm = LLM(
        model=args.model_path,
        dtype="bfloat16",
        tensor_parallel_size=1,
        # gpu_memory_utilization=0.7,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
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
        model_card_text = entry["modelcard"]
        prompt_text = build_prompt(model_card_text)
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
    chat_prompts = convert_to_chat(safe_prompts, tokenizer)
    outputs = llm.generate(chat_prompts, sampling_params)

    # =========================
    # POST-PROCESSING
    # =========================

    for entry, output in zip(collected_models, outputs):
        # model_id = entry["model_id"]
        generated = output.outputs[0].text.strip()
        entry["model_description"] = generated

    # =========================
    # WRITE SINGLE OUTPUT FILE
    # =========================

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(collected_models, f, ensure_ascii=False, indent=2)

    print(f"\nCompleted. Processed {len(collected_models)} model cards.")


if __name__ == "__main__":
    main()
