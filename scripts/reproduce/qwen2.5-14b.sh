#!/usr/bin/env bash
# One-step reproduction of Qwen2.5-14B (Table 4): environment, data, layer-wise curriculum, LoRA recovery stage,
# lm-eval 0.4.3 and the results table, all with the settings of the paper (see ../../reproduce.sh).
#
#   48to35  48 -> 35 layers (25%)   paper AVG 58.6
#   48to24  48 -> 24 layers (50%)   paper AVG 45.7
#
#   bash scripts/reproduce/qwen2.5-14b.sh            # both ratios, one after another
#   bash scripts/reproduce/qwen2.5-14b.sh 48to35     # one ratio
#   GPUS=0,1,2,3 OUT_DIR=/data/outputs bash scripts/reproduce/qwen2.5-14b.sh
#
# Results: <OUT_DIR>/qwen2.5-14b_<ratio>/results.txt (OUT_DIR defaults to outputs/ in the repository root).
set -euo pipefail
[ $# -gt 0 ] || set -- 48to35 48to24
exec bash "$(dirname "${BASH_SOURCE[0]}")/../../reproduce.sh" "${@/#/qwen2.5-14b_}"
