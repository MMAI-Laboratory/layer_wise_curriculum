#!/usr/bin/env bash
# One-step reproduction of LLaMA-3.1-8B (Table 3): environment, data, layer-wise curriculum, LoRA recovery stage,
# lm-eval 0.4.3 and the results table, all with the settings of the paper (see ../../reproduce.sh).
#
#   32to23  32 -> 23 layers (25%)   paper AVG 62.3
#   32to16  32 -> 16 layers (50%)   paper AVG 45.8
#
#   bash scripts/reproduce/llama3.1-8b.sh            # both ratios, one after another
#   bash scripts/reproduce/llama3.1-8b.sh 32to23     # one ratio
#   GPUS=0,1,2,3 OUT_DIR=/data/outputs bash scripts/reproduce/llama3.1-8b.sh
#
# Results: <OUT_DIR>/llama3.1-8b_<ratio>/results.txt (OUT_DIR defaults to outputs/ in the repository root).
set -euo pipefail
[ $# -gt 0 ] || set -- 32to23 32to16
exec bash "$(dirname "${BASH_SOURCE[0]}")/../../reproduce.sh" "${@/#/llama3.1-8b_}"
