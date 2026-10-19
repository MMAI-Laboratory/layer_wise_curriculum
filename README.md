# Layer-wise Curriculum Learning for Efficient LLM Compression

[![arXiv](https://img.shields.io/badge/arXiv-2609.19213-b31b1b.svg)](https://arxiv.org/abs/2609.19213)
[![EMNLP 2026](https://img.shields.io/badge/EMNLP-2026-blue.svg)](https://arxiv.org/abs/2609.19213)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4-ee4c2c.svg)](https://pytorch.org/)

Official implementation of **"Layer-wise Curriculum Learning for Efficient LLM Compression"** (EMNLP 2026).

**Donggeon Lee, Dooyeon Na, Seungmin Oh, Jongbin Ryu**<br>
Ajou University, South Korea

## News

- **2026-10-19**: Code released.

## Overview

We compress an LLM by replacing *segments* of consecutive teacher layers with a single student layer and
transferring the knowledge **layer by layer** instead of through the full model:

1. **Layer-wise knowledge transfer.** Student layer `i` is initialized from teacher layer `s_i` and trained to map
   the hidden state entering teacher layer `s_i` to the hidden state leaving teacher layer `s_{i+1}-1`.
   Only one layer and its cached features live on the GPU, so each layer needs about 1 GB of GPU memory and all
   layers are trained in parallel.
2. **Layer-wise curriculum.** Training proceeds from easy to hard optimization problems. In phase 1 every layer
   learns from teacher features. In each later phase the shallowest active layer is frozen, the student is stacked,
   and the remaining (deeper) layers are re-trained on inputs produced by the student itself, which removes the
   feature misalignment between independently trained layers. The phase lengths follow a static exponential schedule
   (main results) or the **convergence-aware adaptive schedule**, which freezes a layer once its loss stagnates.
3. **Feature caching.** Teacher / student hidden states are cached on disk with an outlier-aware int8 quantization
   and streamed to the layer workers.

The result is a standard Hugging Face model with fewer layers that loads with `AutoModelForCausalLM.from_pretrained`
and is evaluated directly with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

## Contents

- [Quick start: one-step reproduction](#quick-start-one-step-reproduction)
- [Installation](#installation)
- [Data preparation](#data-preparation)
- [Step-by-step pipeline](#step-by-step-pipeline)
- [Reproducing the paper](#reproducing-the-paper)
- [Evaluation](#evaluation)
- [Adding a new model](#adding-a-new-model)
- [Citation](#citation)

## Quick start: one-step reproduction

```bash
huggingface-cli login            # Llama models are gated on the Hugging Face Hub
bash reproduce.sh               
```

`reproduce.sh` creates the conda environment `layer-curriculum` (first run only), samples and tokenizes the
training data, runs the layer-wise curriculum and the LoRA recovery stage with the settings of the paper, evaluates
the model with lm-eval and prints the scores in the format of the paper tables:

```
                     ARC-E  ARC-C   RACE  BoolQ   WnGd   HeSW   PIQA MathQA   COQA   MMLU    AVG
llama3.2-3b_28to19    58.8   35.2   37.5   63.4   64.3   59.8   71.9   25.5   66.7   33.1   51.6
```

Every model has its own one-step script that runs both compression ratios, for example:

```bash
bash scripts/reproduce/llama3.2-3b.sh           # both 28 -> 19 and 28 -> 14 layers
bash scripts/reproduce/llama3.2-3b.sh 28to19    # 28 -> 19 only
```

Other experiments are selected by config name (`configs/experiments/*.yaml`) or path:

```bash
bash reproduce.sh llama3.2-3b_28to14 qwen2.5-7b_28to19
bash reproduce.sh main                                  # every LLaMA / Qwen experiment
GPUS=0,1,2,3 OUT_DIR=/data/outputs bash reproduce.sh    # GPUs and output directory (default: all GPUs, outputs/)
```

Outputs are written to `outputs/<experiment>/`:

```
stacked/          # compressed model (HF format), the result of the curriculum
post_train/       # LoRA adapter of the recovery stage
eval_post_train/  # lm-eval results of the compressed model after the recovery stage
results.txt       # the table above
layers/, logs/    # trained student layers (layer<i>.pt), one log file per step
```

While a step runs, its elapsed time and the last line of its log are printed every minute, e.g.
`[running] phase1_layer14 1h02m | --- [6500/15000] epoch 13: 2/5 min (elapsed/epoch) ---`.

The script can be re-run after an interruption: finished steps are recorded in `outputs/<experiment>/.done` and
skipped. The feature cache of a phase is deleted once its layers are trained.

## Installation

`reproduce.sh` runs these commands on its first call:

```bash
conda create -n layer-curriculum python=3.10 -y && conda activate layer-curriculum
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install flash-attn==2.7.3 --no-build-isolation   # used for all LLM experiments
pip install -e .
```

The experiments were run with Python 3.10, CUDA 11.8, PyTorch 2.4.0, `transformers==4.44.2` and
`accelerate==0.32.0` on 8 x NVIDIA RTX A5000 (24 GB). Please keep the `transformers` version: the layer-wise
training calls decoder layers directly with `position_ids`, which changed in later releases.

## Data preparation

`reproduce.sh` prepares the data when it is missing. All LLM experiments use a random subset of
[SlimPajama-6B](https://huggingface.co/datasets/DKYoon/SlimPajama-6B) tokenized with the teacher's tokenizer into
blocks of 1,024 tokens, sampled with the domain proportions of Sheared-LLaMA: 250k documents for knowledge transfer (about 130k blocks, the 120M tokens of the paper) and 100k
documents for the recovery stage (about 52k blocks, the 60M tokens of the paper).

```bash
# one tokenizer family at a time: llama3 (LLaMA-3.x), llama2 (LLaMA-2), qwen (Qwen2.5)
python scripts/prepare_slimpajama.py --model meta-llama/Llama-3.2-3B --num_data 250000 --out_dir data/slimpajama/llama3
python scripts/prepare_slimpajama.py --model meta-llama/Llama-3.2-3B --num_data 100000 --out_dir data/slimpajama/llama3
```

This writes `data/slimpajama/llama3/random_250000-1024` and `.../random_100000-1024`, the paths used in
`configs/experiments/*.yaml`.

BERT and GPT-2 use 120M tokens each: English Wikipedia and BookCorpus for BERT (mixed 3:1 by tokens, the proportions
of the BERT pre-training corpus; blocks of 512 tokens) and OpenWebText for GPT-2 (blocks of 1,024 tokens):

```bash
python scripts/prepare_corpus.py --corpus wikibook --model google-bert/bert-base-uncased --block_size 512 --out_dir data/bert
python scripts/prepare_corpus.py --corpus openwebtext --model gpt2 --block_size 1024 --out_dir data/gpt2
```

## Step-by-step pipeline

`reproduce.sh` calls `scripts/run_curriculum.py`, which runs every step of the curriculum for one config:

```bash
python scripts/run_curriculum.py --config configs/experiments/llama3.2-3b_28to19.yaml \
    --work_dir outputs/llama3.2-3b_28to19 --gpus 0,1,2,3,4,5,6,7 --post_train --cleanup_features
```

**1. Extract features** (phase 1: teacher features)

```bash
python scripts/extract_features.py --model meta-llama/Llama-3.2-3B --layer_config llama3.2-3b_28to19 \
    --train_file data/slimpajama/llama3/random_250000-1024 --out_dir outputs/demo/features/teacher \
    --num_sample 15000 --quant 8 --use_flash_attn        
```

**2. Train each layer** (one process per layer; run them in parallel on different GPUs)

```bash
for i in 14 15 16 17; do
  CUDA_VISIBLE_DEVICES=$((i-14)) python scripts/train_layer.py --model meta-llama/Llama-3.2-3B \
      --layer_config llama3.2-3b_28to19 --layer_idx $i --feature_dir outputs/demo/features/teacher \
      --out_dir outputs/demo/layers --train_epoch 20 --use_flash_attn &
done; wait
```

**3. Stack the layers into an HF model**

```bash
python scripts/stack_layers.py --model meta-llama/Llama-3.2-3B --layer_config llama3.2-3b_28to19 \
    --layer_dir outputs/demo/layers --save_dir outputs/demo/stacked --use_flash_attn
```

**4. Next phases.** Re-extract features with the stacked student, then train the remaining active
layers with `--student outputs/demo/stacked`(see [Curriculum schedule](#curriculum-schedule)), and stack again.

**5. Recovery stage.** One epoch of LoRA training (rank 16 on the q/k/v/o projections) on 100k SlimPajama
documents. `--global_batch_size` fixes the batch size of the paper independently of the number of GPUs:

```bash
accelerate launch --multi_gpu --num_processes 8 scripts/post_train.py --model outputs/demo/stacked \
    --train_file data/slimpajama/llama3/random_100000-1024 --out_dir outputs/demo --run_name post_train \
    --lora --lora_dropout 0.1 --lr 1e-4 --weight_decay 1e-3 --warmup_ratio 0.005 --adam_eps 1e-5 \
    --train_batch_size 1 --global_batch_size 32 --use_flash_attn
```

## Reproducing the paper

Every experiment has its own config in [`configs/experiments`](configs/experiments). Each config lists the layer
selection, the hyperparameters and the curriculum schedule.

| Model | One-step script | Configs | Paper AVG (25% / 50%) |
|---|---|---|---|
| LLaMA-3.2-3B | `scripts/reproduce/llama3.2-3b.sh` | `llama3.2-3b_28to19`, `_28to14` | 51.5 / 40.7 |
| LLaMA-3.1-8B | `scripts/reproduce/llama3.1-8b.sh` | `llama3.1-8b_32to23`, `_32to16` | 62.3 / 45.8 |
| LLaMA-2-7B | `scripts/reproduce/llama2-7b.sh` | `llama2-7b_32to23`, `_32to16` | 54.9 / 43.4 |
| LLaMA-2-13B | `scripts/reproduce/llama2-13b.sh` | `llama2-13b_40to30`, `_40to20` | 59.8 / 50.5 |
| Qwen2.5-7B | `scripts/reproduce/qwen2.5-7b.sh` | `qwen2.5-7b_28to19`, `_28to14` | 50.9 / 43.0 |
| Qwen2.5-14B | `scripts/reproduce/qwen2.5-14b.sh` | `qwen2.5-14b_48to35`, `_48to24` | 58.6 / 45.7 |

BERT-base and GPT-2, 12 to 6 layers : `bash reproduce.sh bert-base_12to6 gpt2_12to6` prepares the
data and runs the curriculum and the recovery stage (see [BERT and GPT-2](#bert-and-gpt-2)).

Hyperparameters (AdamW; batch size = global batch size):

| | Knowledge transfer | Recovery stage |
|---|---|---|
| LLaMA | batch 1024, lr 3e-4, linear, warm-up 0.005, wd 1e-3, betas (0.9, 0.95) | 1 epoch, LoRA r=16, batch 32 (8B: 64), lr 1e-4 (LLaMA-2-13B: 6e-5), linear, warm-up 0.005, wd 1e-3 |
| Qwen2.5 | as LLaMA, wd 0.1 | 1 epoch, batch 32 (14B: 64), lr 1e-4 (14B: 6e-5), wd 1e-2 |
| BERT | batch 256, lr 1e-4, cosine, warm-up 0.05, wd 1e-2, betas (0.9, 0.999) | 3 epochs MLM, same optimizer settings |
| GPT-2 | batch 512, lr 6e-4, linear, warm-up 0.05, wd 1e-2, betas (0.9, 0.98) | 3 epochs CLM, same optimizer settings |

Key configuration fields:

```yaml
layer_config: llama3.2-3b_28to19   # teacher layers kept (layer_curriculum/layer_selection.py)
extract:
  num_sample: 15000                # cached samples per segment
  quant: 8                         # outlier-aware int8 feature cache (4: int4, 16: fp16)
  gpus_per_model: 1                # GPUs holding one model copy during extraction
train:                             # arguments of scripts/train_layer.py
  lr: 3.0e-4
  warmup_ratio: 0.005
  grad_accum_steps: 1024           # batch size 1 x 1024 accumulation steps
  adaptive: false                  # true: convergence-aware adaptive schedule (delta 1e-4, patience 2, min. 5 epochs)
curriculum:
  alpha: 0.035                     # exponential schedule -> epochs of each phase, see below
```

### Curriculum schedule

`psi(t)`, the index of the shallowest active layer at epoch `t`, follows the static exponential schedule
`psi(t) = exp(alpha * t) + beta` of the main experiments (Appendix C) with `alpha = 0.035` and `beta = 0`
(`psi(0) = 1`). `psi(t)` is discretized as an integer, so phase `p` ends at the first epoch `t` with
`exp(alpha * t) >= p + 1`. `scripts/run_curriculum.py` derives the epochs of every phase from `alpha`
(`layer_curriculum/curriculum.py: schedule_epochs`) and prints them at the start:

| Trainable layers (phases) | Epochs of each phase (`alpha = 0.035`) | Configs |
|---|---|---|
| 3 | 20, 12, 8 | LLaMA-2-13B 40 &rarr; 30 |
| 5 | 20, 12, 8, 6, 6 | BERT-base, GPT-2 12 &rarr; 6 |
| 4 | 20, 12, 8, 6 | all other 25% configs |
| 6 | 20, 12, 8, 6, 6, 4 | LLaMA-2-13B 40 &rarr; 20 |
| 7 | 20, 12, 8, 6, 6, 4, 4 | LLaMA-3.2-3B, Qwen2.5-7B 50% |
| 8 | 20, 12, 8, 6, 6, 4, 4, 3 | LLaMA-3.1-8B, LLaMA-2-7B, Qwen2.5-14B 50% |

A larger `alpha` moves the curriculum faster, and `beta` can also be set in the config.

### Convergence-aware adaptive schedule

The static schedule fixes the curriculum transitions in advance. The convergence-aware adaptive schedule (Appendix C)
instead lets the optimization state of each layer decide when it is frozen. It is enabled in the `train` section of
any config:

```yaml
train:
  adaptive: true                   # convergence-aware adaptive schedule
  adaptive_loss_delta: 1.0e-4      # delta: minimum improvement of the epoch loss
  adaptive_loss_patience: 2        # P: epochs without improvement before a layer is stabilized
  adaptive_min_epoch: 5            # E_min: no layer stops before this epoch
```

Every active layer tracks its layer-wise loss averaged over all mini-batches of an epoch. The reference loss is
refreshed only when the epoch loss improves on it by more than `delta`; otherwise a stagnation counter grows. A layer
is stabilized once it has stagnated for `P` epochs and has trained for at least `E_min` epochs: it stops training,
its weights are saved, and its log reports `Adaptive loss converged at epoch N`. A phase ends once every active
layer is stabilized or has reached the maximum epochs of the phase. The shallowest active layer is then frozen and
the next phase continues from the stacked student, as with the static schedule. The epochs of each phase derived from
`alpha` serve as the maximum epochs `E_max`. All values above are the defaults (`scripts/train_layer.py`), so
`adaptive: true` alone gives the setting of the paper.


## Evaluation

Zero-shot evaluation uses `lm_eval==0.4.3` with `max_length=1024` and batch size 1 (as in `reproduce.sh`):

```bash
lm_eval --model hf --tasks ai2_arc,race,boolq,winogrande,hellaswag,piqa,mathqa,coqa,mmlu --device cuda:0 \
    --model_args pretrained=outputs/llama3.2-3b_28to19/stacked,max_length=1024,peft=outputs/llama3.2-3b_28to19/post_train \
    --batch_size 1 --trust_remote_code --output_path outputs/llama3.2-3b_28to19/eval_post_train
python scripts/summarize_results.py outputs/llama3.2-3b_28to19/eval_post_train   # columns and metrics of Tables 3 / 4
```

The tables report `acc_norm` for ARC-E/C, HellaSwag, PIQA and MathQA, `acc` for RACE, BoolQ, WinoGrande and MMLU,
and F1 for CoQA.


## Adding a new model

The code is architecture-agnostic except for module names. To support a new decoder-only family, add an entry
keyed by `config.model_type` to `ARCHITECTURES` in
[`layer_curriculum/architectures.py`](layer_curriculum/architectures.py) and define a layer selection.
LLaMA and Qwen2 share the same spec.


## Acknowledgements

This code builds on [Hugging Face Transformers](https://github.com/huggingface/transformers),
[Accelerate](https://github.com/huggingface/accelerate) and [PEFT](https://github.com/huggingface/peft). The training
loops are adapted from the Transformers example `run_clm_no_trainer.py`. Evaluation uses
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), and training data is drawn from
[SlimPajama-6B](https://huggingface.co/datasets/DKYoon/SlimPajama-6B).

## Citation

```bibtex
@inproceedings{lee2026layerwise,
  title     = {Layer-wise Curriculum Learning for Efficient {LLM} Compression},
  author    = {Lee, Donggeon and Na, Dooyeon and Oh, Seungmin and Ryu, Jongbin},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026}
}
```

## License

The code is released under the [MIT License](LICENSE). Compressed models derived from LLaMA or Qwen
weights are subject to the licenses of the respective base models.
