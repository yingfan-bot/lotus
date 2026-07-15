# Bridging the Gap Between Latent and Explicit Reasoning with Looped Transformers

This project is for the paper: [Bridging the Gap Between Latent and Explicit Reasoning with Looped Transformers](https://arxiv.org/abs/2606.31779).

<!-- Project Page URL assumes GitHub Pages is enabled (Settings -> Pages -> deploy from branch: main, folder: /docs) on the public yingfan-bot/lotus repo. -->
🔗 **[Project Page](https://yingfan-bot.github.io/lotus/)** &nbsp;·&nbsp; 📄 **[arXiv](https://arxiv.org/abs/2606.31779)** &nbsp;·&nbsp; 🤗 **[Models &amp; collection](https://huggingface.co/collections/yingfanbot/looped-padded-6a552f7ef667cb41db2431a3)**

## Repository layout

```text
.
├── args/                    # YAML training/evaluation configs
├── data/                    # JSON datasets used by the provided configs
├── preprocessing/           # GSM8K download + preprocessing scripts
├── launch_train.sh          # Public torch.distributed launcher
├── environment.yml          # Conda env (Python 3.12, PyTorch 2.7 + CUDA 12.8)
├── requirements.txt         # Python deps layered on top of the NGC image
└── scripts/                 # Python entry points and modules
    ├── run.py               # Main training entry point
    ├── eval.py              # Standalone evaluation entry point
    ├── dataset.py           # Data loading/collation utilities
    ├── lotus.py             # LOTUS latent reasoning model wrapper
    └── utils.py             # Config and seed helpers
```

## Environment

We recommend the NVIDIA NGC PyTorch image (it provides a CUDA-matched PyTorch stack, so
`requirements.txt` does not install `torch`):

```text
nvcr.io/nvidia/pytorch:25.03-py3
```

### Docker

From the repository root:

```bash
docker run --gpus all --rm -it \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$PWD":/workspace/lotus \
  -w /workspace/lotus \
  nvcr.io/nvidia/pytorch:25.03-py3

pip install -r requirements.txt
```

### Conda environment

If you are not using a container, an `environment.yml` is provided that mirrors the NGC
image (Python 3.12, PyTorch 2.7 + CUDA 12.8). From the repository root:

```bash
conda env create -f environment.yml
conda activate lotus
```

## Authentication

If you need private Hugging Face models or W&B logging, log in before launching:

```bash
huggingface-cli login
wandb login
```

The launcher itself does not handle credentials.

## Data

The configs read JSON splits of the form `{"question", "steps", "answer"}` from `./data/`
(git-ignored, not shipped). To download and preprocess the augmented-CoT GSM8K splits into
that format, run from the repo root:

```bash
bash preprocessing/gsm_icot.bash
```

This produces `data/gsm_{train,valid,test}.json`. The scripts under
[`preprocessing/`](preprocessing) are adapted from
[Coconut](https://github.com/facebookresearch/coconut) and pull the augmented GSM8K data
from [Internalize_CoT_Step_by_Step](https://github.com/da03/Internalize_CoT_Step_by_Step).

## Configuration

Each run is configured by a YAML file passed to `scripts/run.py` (via `CONFIG`). Configs compose
through a `base:` chain under [`args/base/`](args/base) (shared defaults → model → method
→ dataset), so each leaf config in [`args/`](args) only overrides what it needs, and every
field is documented inline in the YAML. The essentials:

- `looped` — `True` for LOTUS (looped latent training); `cot: True` for plain CoT fine-tuning.
- `load_model_path` — checkpoint (HF repo id or local path) to initialize LOTUS from a CoT-tuned model, or to evaluate.
- `c_thought` / `max_latent_stage` / `epochs_per_stage` — latent-token count and curriculum schedule.

## Models

Trained checkpoints are on the Hugging Face Hub — collected in the
[**LOTUS collection**](https://huggingface.co/collections/yingfanbot/looped-padded-6a552f7ef667cb41db2431a3)
(paper + models):

| Model | Description | GSM8K (GSM8k-Aug) |
| --- | --- | --- |
| [`yingfanbot/gsm-lotus-llama3b`](https://huggingface.co/yingfanbot/gsm-lotus-llama3b) | LOTUS, Llama-3.2-3B | 70.05% |
| [`yingfanbot/gsm-lotus-llama3b-codi`](https://huggingface.co/yingfanbot/gsm-lotus-llama3b-codi) | LOTUS + CODI, Llama-3.2-3B | 70.58% |

Stage-1 CoT initialization checkpoints:
[`gsm-cot-gpt2`](https://huggingface.co/yingfanbot/gsm-cot-gpt2),
[`gsm-cot-llama1b`](https://huggingface.co/yingfanbot/gsm-cot-llama1b),
[`gsm-cot-llama3b`](https://huggingface.co/yingfanbot/gsm-cot-llama3b).

## Training

Use `launch_train.sh` for distributed training. Important environment variables:

| Variable | Default | Description |
| --- | --- | --- |
| `CONFIG` | `args/gsm8k_cot_llama1b.yaml` | YAML config passed to `scripts/run.py` |
| `RUN_NAME` | empty | Optional run-name override |
| `NPROC_PER_NODE` | `4` | GPUs per node; set to `128 / batch_size_training` so the overall batch is 128 (GPT-2: 2, Llama-1B: 4, Llama-3B: 8) |
| `MASTER_PORT` | `29500` | Torch distributed master port |

LOTUS is trained in **two stages**: (1) CoT supervised fine-tuning, then (2) looped
latent training initialized from the Stage-1 checkpoint. Setting `RUN_NAME` controls
the output directory (`./outputs/<RUN_NAME>/`), which keeps the wiring between the two
stages predictable.

> By default, the LOTUS configs initialize from the published CoT checkpoints on Hugging Face
> (`yingfanbot/gsm-cot-{gpt2,llama1b,llama3b}`), so Stage 2 can be run directly without first
> running Stage 1. To initialize from your own CoT checkpoint instead, set `load_model_path`
> (and `teacher_model_path`, if CODI distillation is enabled) to
> `./outputs/<stage-1 RUN_NAME>/checkpoint_final`.

**GPT-2**

```bash
# Stage 2 — LOTUS (loads the published GPT-2 CoT checkpoint by default)
CONFIG=args/gsm8k_lotus_gpt2.yaml RUN_NAME=gsm-lotus-gpt2 NPROC_PER_NODE=2 ./launch_train.sh

# (optional) retrain the CoT stage yourself, then point load_model_path at the output:
# CONFIG=args/gsm8k_cot_gpt2.yaml RUN_NAME=gsm-cot-gpt2 NPROC_PER_NODE=2 ./launch_train.sh
#   -> ./outputs/gsm-cot-gpt2/checkpoint_final
```

**Llama-3.2-1B**

```bash
# Stage 2 — LOTUS (loads the published Llama-1B CoT checkpoint by default)
CONFIG=args/gsm8k_lotus_llama1b.yaml RUN_NAME=gsm-lotus-llama1b NPROC_PER_NODE=4 ./launch_train.sh

# (optional) retrain the CoT stage yourself, then point load_model_path at the output:
# CONFIG=args/gsm8k_cot_llama1b.yaml RUN_NAME=gsm-cot-llama1b NPROC_PER_NODE=4 ./launch_train.sh
#   -> ./outputs/gsm-cot-llama1b/checkpoint_final
```

**Llama-3.2-3B**

```bash
# Stage 2 — LOTUS (loads the published Llama-3B CoT checkpoint by default)
CONFIG=args/gsm8k_lotus_llama3b.yaml RUN_NAME=gsm-lotus-llama3b NPROC_PER_NODE=8 ./launch_train.sh

# (optional) retrain the CoT stage yourself, then point load_model_path (+ teacher_model_path) at the output:
# CONFIG=args/gsm8k_cot_llama3b.yaml RUN_NAME=gsm-cot-llama3b NPROC_PER_NODE=4 ./launch_train.sh
#   -> ./outputs/gsm-cot-llama3b/checkpoint_final
```

## Evaluation

`scripts/eval.py` is the standalone evaluation script (single GPU). It can load weights two ways:

- **Local checkpoint** — pass `--checkpoint ./outputs/<run>/checkpoint_final` (a torch state dict
  produced by training) together with the matching `--model_id` base model.
- **Published HF model** — pass `--model_id <hf-repo>` and omit `--checkpoint`; the trained weights load
  straight from the repo's `safetensors`. `from_pretrained` loads the weights only — the looped padded
  architecture is provided by the `Lotus` wrapper (`eval.py` runs it), so no separate checkpoint file is needed.

**LOTUS model on GSM8K (local checkpoint):**

```bash
python scripts/eval.py \
  --checkpoint ./outputs/gsm-lotus-llama3b/checkpoint_final \
  --model_id meta-llama/Llama-3.2-3B-Instruct \
  --datasets gsm8k \
  --n_looped_iters 6 --c_thought 25 --bf16
```

**LOTUS model on GSM8K (published HF model — no checkpoint file):**

```bash
python scripts/eval.py \
  --model_id yingfanbot/gsm-lotus-llama3b \
  --datasets gsm8k \
  --n_looped_iters 6 --c_thought 25 --bf16
```

**LOTUS model on out-of-distribution sets** (GSM-Hard / MultiArith / SVAMP):

```bash
python scripts/eval.py \
  --checkpoint ./outputs/gsm-lotus-llama3b/checkpoint_final \
  --datasets gsm-hard multi-arith svamp \
  --n_looped_iters 6 --c_thought 25 --bf16 \
  --save_preds preds.json
```

**Plain CoT model** (add `--cot`):

```bash
python scripts/eval.py \
  --checkpoint ./outputs/gsm-cot-llama3b/checkpoint_final \
  --datasets gsm8k --cot --bf16
```

Match `--n_looped_iters` / `--c_thought` to how the checkpoint was trained — use `--c_thought 13`
for GPT-2 checkpoints and `--c_thought 25` for Llama-1B/3B (and set `--model_id` to the matching
base model).

## Citation

If you use this code, please cite LOTUS:

```bibtex
@article{fan2026bridging,
  title={Bridging the Gap Between Latent and Explicit Reasoning with Looped Transformers},
  author={Fan, Ying and Svete, Anej and Lee, Kangwook},
  journal={arXiv preprint arXiv:2606.31779},
  year={2026}
}
```

## License

Released under the MIT License (see [`LICENSE`](LICENSE)).

## Acknowledgements

The training, evaluation, and data-preprocessing code is adapted from
[Coconut](https://github.com/facebookresearch/coconut)
(*Training Large Language Models to Reason in a Continuous Latent Space*).
