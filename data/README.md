# Data Pipeline

This folder contains the scripts used to prepare datasets and build the model index files consumed by the rest of the project.

## Overview

The main pipeline lives in [process_data.py](process_data.py). It:

- loads environment variables from the repository root `.env` file
- expects `HF_API_TOKEN` to authenticate with Hugging Face
- sets local cache directories under `.hf_cache/`
- reads raw data from `data/raw/`
- writes processed JSONL files to `data/processed/`

## Before You Run

Make sure you have:

- created and activated the project environment
- set `HF_API_TOKEN` in the repository root `.env` file
- downloaded or placed the required raw datasets under `data/raw/`

## Run Data Processing

The project exposes the data pipeline through the `process-data` command defined in `pyproject.toml`.

From the repository root, run:

```bash
process-data --data <dataset>
```

Supported values for `<dataset>` are:

- `apibench-hf`
- `mllm`
- `hugging-bench-1`
- `hugging-bench-2`

Example:

```bash
process-data --data mllm
```

If you prefer to run the script directly, the equivalent command is:

```bash
HF_API_TOKEN=your_token_here python data/process_data.py --data mllm
```

## Outputs

For each dataset split, the pipeline writes JSONL files in `data/processed/`:

- `cleaned-<dataset>-train.json`
- `cleaned-<dataset>-val.json`
- `cleaned-<dataset>-eval.json`

The training split is created from the first returned dataframe in the processing function, using a stratified train/validation split on `model_name`.

## Generate Model Indices (for retrieval)

This project builds cumulative "model indices" used by retrieval components. Each index is a newline-delimited JSONL file listing the unique model entries observed in one or more experiences. The generator `data/generate_model_indices.py` reads the processed training files and writes cumulative indices named `e1.json`, `e1_e2.json`, `e1_e2_e3.json`, etc., into `data/model_indices/`.

Behavior summary:

- `e1.json` contains models from experience 1 only.
- `e1_e2.json` contains the combined unique models from experiences 1 and 2.

Usage (from the repository root):
After processing the datasets, generate the model index files used by the project by running

```bash
cd data
python generate_model_indices.py
```

See `data/generate_model_indices.py` for implementation details and the exact input filenames expected in `data/processed/`.