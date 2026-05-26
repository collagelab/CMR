# HuggingGPTstyle

This folder contains two scripts:

- `huggingQ.py`: runs vLLM on a JSONL dataset and saves model outputs.
- `eval_huggingQ.py`: evaluates outputs against a ground-truth dataset.


## Setup

Do NOT use the repository-root virtual environment ([CARvE README](../../../README.md)). HuggingGPTstyle requires the `self-instruct/.venv` virtual environment. See [self-instruct README](../../../self-instruct/README.md) for full setup instructions.


All commands below should be run from the repository root.

## Run inference

`huggingQ.py` expects a JSONL dataset (one JSON per line) with `instruction` and `domain` fields.


```bash
# apibench
python core/baselines/HuggingGPTstyle/huggingQ.py \
  --dataset_path data/processed/cleaned-apibench-hf-eval.json \
  --output_dir core/baselines/HuggingGPTstyle/results \
  --output_file_name huggingQ_apibench_test.json \
  --model_name Qwen/Qwen3-32B

# mllm
python core/baselines/HuggingGPTstyle/huggingQ.py \
  --dataset_path data/processed/cleaned-mllm-eval.json \
  --output_dir core/baselines/HuggingGPTstyle/results \
  --output_file_name huggingQ_mllm_test.json \
  --model_name Qwen/Qwen3-32B

# hugging-bench-1
python core/baselines/HuggingGPTstyle/huggingQ.py \
  --dataset_path data/processed/cleaned-hugging-bench-1-eval.json \
  --output_dir core/baselines/HuggingGPTstyle/results \
  --output_file_name huggingQ_hugging_bench_1_test.json \
  --model_name Qwen/Qwen3-32B

# hugging-bench-2
python core/baselines/HuggingGPTstyle/huggingQ.py \
  --dataset_path data/processed/cleaned-hugging-bench-2-eval.json \
  --output_dir core/baselines/HuggingGPTstyle/results \
  --output_file_name huggingQ_hugging_bench_2_test.json \
  --model_name Qwen/Qwen3-32B

```



## Evaluate

`eval_huggingQ.py` compares model outputs to ground truth and prints accuracy.


```bash
# apibench
python core/baselines/HuggingGPTstyle/eval_huggingQ.py \
  --input_file core/baselines/HuggingGPTstyle/results/huggingQ_apibench_test.json \
  --dataset_file data/processed/cleaned-apibench-hf-eval.json

# mllm
python core/baselines/HuggingGPTstyle/eval_huggingQ.py \
  --input_file core/baselines/HuggingGPTstyle/results/huggingQ_mllm_test.json \
  --dataset_file data/processed/cleaned-mllm-eval.json

# hugging-bench-1
python core/baselines/HuggingGPTstyle/eval_huggingQ.py \
  --input_file core/baselines/HuggingGPTstyle/results/huggingQ_hugging_bench_1_test.json \
  --dataset_file data/processed/cleaned-hugging-bench-1-eval.json

# hugging-bench-2
python core/baselines/HuggingGPTstyle/eval_huggingQ.py \
  --input_file core/baselines/HuggingGPTstyle/results/huggingQ_hugging_bench_2_test.json \
  --dataset_file data/processed/cleaned-hugging-bench-2-eval.json
```

## Outputs

- Inference results are saved under `core/baselines/HuggingGPTstyle/results/`.
- Evaluation prints accuracy to stdout.
