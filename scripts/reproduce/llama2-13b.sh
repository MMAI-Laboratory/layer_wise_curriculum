#!/usr/bin/env bash
# One-step reproduction of LLaMA-2-13B (Table 3): environment, data, layer-wise curriculum, LoRA recovery stage,
# lm-eval 0.4.3 and the results table, all with the settings of the paper (see ../../reproduce.sh).
#
#   40to30  40 -> 30 layers (25%)   paper AVG 59.8
#   40to20  40 -> 20 layers (50%)   paper AVG 50.5
#
#   bash scripts/reproduce/llama2-13b.sh            # both ratios, one after another
#   bash scripts/reproduce/llama2-13b.sh 40to30     # one ratio
#   GPUS=0,1,2,3 OUT_DIR=/data/outputs bash scripts/reproduce/llama2-13b.sh
#
# Results: <OUT_DIR>/llama2-13b_<ratio>/results.txt (OUT_DIR defaults to outputs/ in the repository root).
set -euo pipefail
[ $# -gt 0 ] || set -- 40to30 40to20
exec bash "$(dirname "${BASH_SOURCE[0]}")/../../reproduce.sh" "${@/#/llama2-13b_}"
