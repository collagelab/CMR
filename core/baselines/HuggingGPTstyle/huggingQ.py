import argparse
import json
import os

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import os


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


def construct_prompts(path_dataset):
    print("LOAD DATASET")

    dataset_json = []
    with open(path_dataset, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dataset_json.append(json.loads(line))

    # Filter out None domains before sorting
    if None in set(elm["domain"] for elm in dataset_json if elm.get("domain")):
        print(
            "Warning: Some entries have 'domain' set to None. These will be excluded from the domain list."
        )

    domain_list = sorted(
        set(elm["domain"] for elm in dataset_json if elm.get("domain") is not None)
    )
    domain_str = ", ".join(domain_list)

    base_prompt = (
        "You are given an ###Instruction\n"
        "Your task is to identify which domain it belongs to.\n\n"
        f"###Possible_domains \n{domain_str}\n\n"
        "###Output_format\n"
        "- Return ONLY the domain name, exactly as written in the list above.\n"
        "- Do NOT include explanations or extra text.\n"
        "- If uncertain, choose the closest matching domain.\n\n"
    )
    # if elm['instruction'] not start with ###Instruction: add it
    for elm in dataset_json:
        if not elm["instruction"].startswith("###Instruction"):
            elm["instruction"] = f"###Instruction:\n{elm['instruction']}"
    prompts = [
        f"{base_prompt}{elm['instruction']}\n\n###Response:\n" for elm in dataset_json
    ]
    return prompts


def main():
    parser = argparse.ArgumentParser(description="Generate LLM outputs with vLLM.")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=False,
        default="../data/processed/cleaned-hugging-bench-1-eval.json",
        help="Path to the JSONL dataset (default: ../data/processed/cleaned-hugging-bench-1-eval.json)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default="./results",
        help="Output directory for JSON results (default: ./results)",
    )
    parser.add_argument(
        "--output_file_name",
        type=str,
        default="results.json",
        help="Output JSON filename",
    )
    #parse model name 
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-32B",
        help="Model name or path (default: Qwen/Qwen3-32B)",
    )
    args = parser.parse_args()
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hugging_prompts = construct_prompts(args.dataset_path)
    hugging_prompts = convert_to_chat(
        hugging_prompts, tokenizer=tokenizer, enable_thinking=False
    )

    llm = LLM(
        model=model_name,
        max_model_len=4000,
        tensor_parallel_size=1,
        #gpu_memory_utilization=0.7,
    )

    # Definizione dei parametri di sampling
    sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=2048)

    print(f"Running inference on {len(hugging_prompts)} prompts...")

    # Generazione di tutti i prompt senza gestire batch manualmente
    outputs = llm.generate(hugging_prompts, sampling_params=sampling_params)

    results = []
    for i, gen_result in enumerate(outputs):
        results.append(
            {
                "prompt": hugging_prompts[i],
                "generated_text": [c.text for c in gen_result.outputs],
            }
        )

    # Ensure the output directory exists (args.output_dir is a directory)
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, args.output_file_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved results in {out_path}")


if __name__ == "__main__":
    main()
