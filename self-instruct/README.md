# Self-Instruct Pipeline

This folder contains the data collection and self-instruct pipeline used to create HuggingBench.

The pipeline has two branches: the recent flow processes newer Hugging Face model cards, while the legacy flow processes models in previous experiences (ApiBench and ToolMllm). Both branches follow the same cleaning and prompt-generation steps, but they run on different input sets and use the .env variable `LEGACY_MODELS` to select the branch.

## Self-Instruct virtual environment 

Use a dedicated virtual environment at `self-instruct/.venv` to isolate dependencies from the main project.
Do NOT use the repository-root virtual environment ([CARvE README](../README.md)) for any self-instruct steps — create and use only `self-instruct/.venv`.
Do not install `self-instruct` with `-e ..` into the main environment; install dependencies only inside `self-instruct/.venv`, because pinned packages in the main project (such as `torch==2.8.0`) may conflict.

From the repository root:

```bash
# If uv is not installed, install it once:
curl -LsSf https://astral.sh/uv/install.sh | sh
```


```bash
# Create the self-instruct environment (one-time setup)
uv venv --python 3.11 self-instruct/.venv

# Activate it before running any self-instruct steps
source self-instruct/.venv/bin/activate

# Install self-instruct-only dependencies
uv pip install -r self-instruct/requirements.txt

# Register the self-instruct kernel for notebooks
python -m ipykernel install --user --name self-instruct --display-name "Python (self-instruct)"
```

When you want to return to the main project environment, run `deactivate` and then activate your root environment again. If you accidentally run steps inside the main project's venv, stop and recreate `self-instruct/.venv` before continuing.

## Run 

Follow these steps from the repository root. The key points to note: create `self-instruct/.env` once, and for the manual-cleaning notebooks (Steps 4 and 6) you must change `LEGACY_MODELS` in `self-instruct/.env` between runs and restart the notebook kernel.

0) Create `self-instruct/.env` (one-time)

```bash
cat > self-instruct/.env << 'EOF'
LEGACY_MODELS=0
EOF
```

1) Download recent model cards (Step 1)

Open `self-instruct/code/step_1_download_model_hub.ipynb` and run all cells in order.

2) Download legacy model cards (Step 2)

Open `self-instruct/code/step_2_download_model_hub_legacy_models.ipynb` and run all cells in order.

3) Clean model-card descriptions (Step 3)

Use `Qwen/Qwen3-32B` as the reference local model path.

New models:
```bash
python self-instruct/code/step_3_clean_model_cards.py \
	--model_cards_path self-instruct/data/model_cards_step_1.jsonl \
	--output_path self-instruct/data/cleaned_model_cards_step_3.jsonl \
	--model_path Qwen/Qwen3-32B
```

Legacy models:
```bash
python self-instruct/code/step_3_clean_model_cards.py \
	--model_cards_path self-instruct/data/legacy_model_cards_from_step_2.jsonl \
	--output_path self-instruct/data/cleaned_model_cards_legacy_step_3.jsonl \
	--model_path Qwen/Qwen3-32B
```

4) Manual cleaning of model cards (Step 4) — run twice with a mid-run env change

Open `self-instruct/code/step_4_manual_cleaning_model_cards.ipynb` and run all cells for the recent pass (with `LEGACY_MODELS=0`).

When that completes, switch to the legacy pass by overwriting the `.env` and restarting the notebook kernel, then re-run all cells:

```bash
# switch to legacy mode (overwrite .env)
echo 'LEGACY_MODELS=1' > self-instruct/.env
# In Jupyter: Kernel -> Restart, then run all cells again
```

5) Generate prompts (Step 5)

Use `Qwen/Qwen3-32B` as the reference local model path.

```bash
# recent
python self-instruct/code/step_5_self_instruct.py \
	--model_cards_path self-instruct/data/cleaned_model_cards_step_4.jsonl \
	--output_path self-instruct/data/self_instructed_models_step_5.jsonl \
	--model_path Qwen/Qwen3-32B

# legacy
python self-instruct/code/step_5_self_instruct.py \
	--model_cards_path self-instruct/data/cleaned_model_cards_legacy_step_4.jsonl \
	--output_path self-instruct/data/self_instructed_legacy_models_step_5.jsonl \
	--model_path Qwen/Qwen3-32B
```

6) Clean generated prompts (Step 6) — run twice with a mid-run env change

Open `self-instruct/code/step_6_manual_cleaning_prompts_self_instruct.ipynb` and run all cells for the recent pass (with `LEGACY_MODELS=0`).

When that completes, switch to the legacy pass as in Step 4:

```bash
echo 'LEGACY_MODELS=1' > self-instruct/.env
# In Jupyter: Kernel -> Restart, then run all cells again
```

7) Build experiences and split prompts into exp3/exp4 (Step 7)

Open `self-instruct/code/step_7_create_exp_3_4.ipynb` and run all cells in order.




