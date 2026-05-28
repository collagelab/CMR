# Continual Model Routing in Evolving Model Hubs

<p align="center">
  <a href="https://arxiv.org/abs/2605.28577"><img src="https://img.shields.io/badge/ArXiv-Paper-brown" alt="Paper"></a>
  <a href="https://github.com/collagelab/CMR/"><img src="https://img.shields.io/badge/GitHub-Code-orange" alt="Code"></a>
  <a href="https://www.vincenzolomonaco.com/collage-lab"><img src="https://img.shields.io/badge/Project-Page-purple" alt="Lab"></a>
</p>

<p align="center">
<b><a href="https://scholar.google.com/citations?user=MMukC9kAAAAJ&hl=en">Jack Bell</a></b>, <b><a href="https://scholar.google.com/citations?user=EvXdzIIAAAAJ&hl=it&oi=ao">Giacomo Carfi</a></b>, <a href="https://scholar.google.com/citations?user=SrB4KocAAAAJ&hl=it&oi=ao">Gerlando Gramaglia</a>, <a href="https://www.vincenzolomonaco.com/">Vincenzo Lomonaco</a>
</p>

<p align="center">
<b>University of Pisa</b> &nbsp;·&nbsp; <b>Luiss University</b><br>
</p>

<p align="center">
<img src="https://raw.githubusercontent.com/collagelab/CMR/main/assets/cmr_concept.jpg" width="100%"/>
</p>

AI model hubs now host millions of pre-trained models spanning modalities, domains, and levels of specialisation. A practical system must select the appropriate model for a given query before inference, without executing multiple candidates, and must do so reliably as the hub continues to grow. This repository accompanies our ICML 2026 paper, which formalises this setting as Continual Model Routing (CMR): a class-incremental learning problem where a router must incorporate newly released models over time while retaining competence on those it has already learned to route.

<p align="center">
<img src="https://raw.githubusercontent.com/collagelab/CMR/main/assets/carve_method.png" width="100%"/>
</p>

The paper makes three main contributions:

- CMRBench, a large-scale benchmark spanning four sequential experiences, over 2,000 candidate models, and multiple domains (APIBench, ToolMMBench, and our new HuggingBench), constructed to simulate realistic hub expansion.
- Model family accuracy, a new evaluation metric that occupies a middle ground between exact model-ID matching and coarse domain-level accuracy, capturing functional similarity across model variants.
- CARvE (Continual Anchored Router via contrastive Embeddings), an embedding-based router that scales to thousands of candidate models through fixed-size candidate-set training and a persistent model registry, with checkpoint-based anchoring to limit catastrophic forgetting across experiences.

With 10% replay, CARvE achieves 80.7% domain accuracy and 5.9% forgetting, compared to 75.9% and 13.1% for standard replay, while requiring roughly 45% fewer FLOPs than cumulative retraining.

# CARvE

This is the main entry point for running CARvE and setting up the project.

## Documentation Map

- Main CARvE guide: `README.md` (this page)
- Main baselines guide: [core/baselines/README.md](core/baselines/README.md)
- HuggingGPT baseline guide: [core/baselines/HuggingGPTstyle/README.md](core/baselines/HuggingGPTstyle/README.md)
- Data generation guide: [data/README.md](data/README.md)
- Self-instruct pipeline guide: [self-instruct/README.md](self-instruct/README.md)

## Project Structure

```text
CMR/
├── core/                             # Main Python package
│   ├── baselines/                    # Baseline training/eval code
│   │   ├── main.py                   # Baseline train entrypoint
│   │   └── eval_continual.py         # Baseline eval entrypoint
│   ├── carve/                        # CARvE training/eval code
│   │   ├── main_carve.py             # CARvE train entrypoint
│   │   └── eval_carve.py             # CARvE eval entrypoint
│   └── experiments/                  # Checkpoints and experiment artifacts
├── configurations_carve/             # CARvE YAML configs
├── configurations/                   # Baseline YAML configs
├── data/
│   ├── raw/                          # Raw datasets
│   ├── processed/                    # Processed JSON files
│   └── process_data.py               # Data preparation script
├── batch_training_logs/              # Batch/ablation logs
├── results/                          # Evaluation outputs and aggregated metrics
└── pyproject.toml                    # Dependencies + CLI entrypoints
```

## Initial Setup

1) Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2) Create environment and install:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

3) Configure environment variables:

```bash
cp .env.example .env
```

Set at least:

- `HF_API_TOKEN` for dataset processing and Hugging Face downloads
- `WANDB_API_KEY` if you log runs to Weights & Biases

## Data Preparation

To know information about how to generate data please read [data/README.md](data/README.md).

## Run CARvE

### 1) First experience (starter config)

```bash
train-carve --config configurations_carve/apibench.yaml
```

### 2) Continue with later experiences

Use `configurations_carve/mllm_onwards.yaml` and update the key fields:

- `experience_name`
- `lora_adapters`
- router registry paths (when extending registry across experiences)

Then run:

```bash
train-carve --config configurations_carve/mllm_onwards.yaml
```

### 3) Evaluate CARvE checkpoints

```bash
eval-carve --config configurations_carve/eval_config.yaml
```

Adjust `experience_name`, `lora_adapters`, and decoding parameters in the eval config as needed.

### Example: full 4-experience sequential run

Run experience 1 with `apibench.yaml`, then resume with `mllm_onwards.yaml` for experiences 2-4.

```bash
# Step 1: experience 1 (apibench)
train-carve --config configurations_carve/apibench.yaml

# Step 2: experiences 2-4 (resume from apibench checkpoint)
train-carve \
  --config configurations_carve/mllm_onwards.yaml \
  --lora_adapters apibench-test_variant/checkpoint-310 \
  --experiences_sequence mllm hugging-bench-1 hugging-bench-2
```

## Outputs

- Checkpoints and adapters: `core/experiments/`
- Logs: `batch_training_logs/`
- Evaluation outputs: `results/`

## Baselines

Baseline training/evaluation instructions are documented in `core/baselines/README.md`.

## Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{continual-router-2026,
  title={Continual Model Routing in Evolving Model Hubs},
  author={Jack Bell, Giacomo Carfi, Gerlando Gramaglia, Vincenzo Lomonaco},
  booktitle={Forty-third International Conference on Machine Learning, {ICML} 2026, Seoul, South Korea, July 6-11},
  year={2026}
}
```
