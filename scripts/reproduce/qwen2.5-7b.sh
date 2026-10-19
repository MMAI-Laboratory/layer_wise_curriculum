#!/usr/bin/env bash
# One-step reproduction of Qwen2.5-7B (Table 4): environment, data, layer-wise curriculum, LoRA recovery stage,
# lm-eval 0.4.3 and the results table, all with the settings of the paper (see ../../reproduce.sh).
#
#   28to19  28 -> 19 layers (25%)   paper AVG 50.9
#   28to14  28 -> 14 layers (50%)   paper AVG 43.0
#
#   bash scripts/reproduce/qwen2.5-7b.sh            # both ratios, one after another
#   bash scripts/reproduce/qwen2.5-7b.sh 28to19     # one ratio
#   GPUS=0,1,2,3 OUT_DIR=/data/outputs bash scripts/reproduce/qwen2.5-7b.sh
#
# Results: <OUT_DIR>/qwen2.5-7b_<ratio>/results.txt (OUT_DIR defaults to outputs/ in the repository root).
set -euo pipefail
[ $# -gt 0 ] || set -- 28to19 28to14
exec bash "$(dirname "${BASH_SOURCE[0]}")/../../reproduce.sh" "${@/#/qwen2.5-7b_}"
