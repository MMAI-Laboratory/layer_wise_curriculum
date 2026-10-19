#!/usr/bin/env bash
# One-step reproduction: environment -> data -> layer-wise curriculum -> recovery stage -> evaluation -> table.
#
#   bash reproduce.sh                                        # LLaMA-3.2-3B, 28 -> 19 layers (Table 3)
#   bash reproduce.sh llama3.2-3b_28to14 qwen2.5-7b_28to19   # config names in configs/experiments or paths
#   bash reproduce.sh main                                   # all LLaMA / Qwen experiments of Tables 3 and 4
#   bash scripts/reproduce/<model>.sh                        # both ratios of one model, e.g. llama3.1-8b, qwen2.5-14b
#   bash reproduce.sh bert-base_12to6 gpt2_12to6             # BERT / GPT-2 (Tables 1, 2): data, curriculum, recovery
#
# Environment variables (all optional):
#   GPUS=0,1,2,3      GPUs to use (default: all GPUs)
#   OUT_DIR=outputs   results go to <OUT_DIR>/<experiment>/ (the feature cache needs up to ~400 GB there)
#   ENV_NAME=layer-curriculum   conda environment created on the first run; SETUP=0 uses the active Python instead
#   JOBS_PER_GPU=1    layer-training processes sharing one GPU
#
# Llama models are gated on the Hugging Face Hub: run `huggingface-cli login` once before.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

OUT_DIR=${OUT_DIR:-outputs}
ENV_NAME=${ENV_NAME:-layer-curriculum}
JOBS_PER_GPU=${JOBS_PER_GPU:-1}
TASKS=ai2_arc,race,boolq,winogrande,hellaswag,piqa,mathqa,coqa,mmlu  
MAIN=(llama3.2-3b_28to19 llama3.2-3b_28to14 llama3.1-8b_32to23 llama3.1-8b_32to16 llama2-7b_32to23 llama2-7b_32to16
      llama2-13b_40to30 llama2-13b_40to20 qwen2.5-7b_28to19 qwen2.5-7b_28to14 qwen2.5-14b_48to35 qwen2.5-14b_48to24)

# ---------------------------------------------------------------- 1. environment (Python 3.10, CUDA 11.8, PyTorch 2.4.0)
setup_env() {
  [ "${SETUP:-1}" = 0 ] && return
  command -v conda > /dev/null || { echo "conda not found: install Miniconda or run with SETUP=0"; exit 1; }
  set +u; eval "$(conda shell.bash hook)"
  if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    conda create -y -n "${ENV_NAME}" -c conda-forge --override-channels python=3.10
  fi
  conda activate "${ENV_NAME}"; set -u
  export PYTHONNOUSERSITE=1  # ignore packages installed in ~/.local
  if [ ! -f "${CONDA_PREFIX}/.layer_curriculum_ready" ]; then
    pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu118
    pip install -r requirements.txt
    pip install flash-attn==2.7.3 --no-build-isolation
    pip install -e .
    touch "${CONDA_PREFIX}/.layer_curriculum_ready"
  fi
}

config_path() {
  for path in "$1" "configs/experiments/$1.yaml"; do
    [ -f "${path}" ] && { echo "${path}"; return; }
  done
  echo "unknown experiment: $1" >&2; exit 1
}

config_value() {  # <config> <key> [<section>]
  python -c "import sys, yaml; c = yaml.safe_load(open(sys.argv[1])); c = c.get(sys.argv[3], {}) if len(sys.argv) > 3 else c
print(c.get(sys.argv[2]) or '')" "$@"
}

# ---------------------------------------------------------------- 2. data (LLMs: SlimPajama-6B with the Sheared-LLaMA
#                                                                     mixture; BERT: Wikipedia + BookCorpus; GPT-2: OpenWebText)
prepare_data() {
  local cfg=$1 model path
  model=$(config_value "${cfg}" model)
  for path in "$(config_value "${cfg}" train_file)" "$(config_value "${cfg}" train_file post_train)"; do
    [ -z "${path}" ] || [ -e "${path}" ] && continue
    if [[ $(basename "${path}") =~ ^random_([0-9]+)-([0-9]+)$ ]]; then
      python scripts/prepare_slimpajama.py --model "${model}" --num_data "${BASH_REMATCH[1]}" \
        --block_size "${BASH_REMATCH[2]}" --out_dir "$(dirname "${path}")"
    elif [[ $(basename "${path}") =~ ^(wikibook|openwebtext)_([0-9]+)M-([0-9]+)$ ]]; then
      python scripts/prepare_corpus.py --corpus "${BASH_REMATCH[1]}" --model "${model}" \
        --num_tokens "${BASH_REMATCH[2]}000000" --block_size "${BASH_REMATCH[3]}" --out_dir "$(dirname "${path}")"
    else
      echo "${cfg}: training data ${path} not found"; exit 1
    fi
  done
}

# ---------------------------------------------------------------- 4. zero-shot evaluation (lm-eval 0.4.3)
evaluate() {  # <gpus> <model_dir> <output_dir> [<lora_adapter_dir>]; several GPUs hold one model copy
  local parallel=""
  [[ $1 == *,* ]] && parallel=",parallelize=True"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$1 lm_eval --model hf \
    --model_args "pretrained=$2,max_length=1024${parallel}${4:+,peft=$4}" --tasks "${TASKS}" --device cuda:0 \
    --batch_size 1 --trust_remote_code --output_path "$3" > "$3.log" 2>&1 &
  local pid=$! start=${SECONDS} last=${SECONDS}
  while kill -0 "${pid}" 2> /dev/null; do  # progress line every minute, as in run_curriculum.py
    sleep 5
    if (( SECONDS - last >= 60 )) && kill -0 "${pid}" 2> /dev/null; then
      last=${SECONDS}
      echo "[running] evaluation $(( (SECONDS - start) / 60 ))m | $(tail -c 4096 "$3.log" | tr '\r' '\n' | grep -v '^\s*$' | tail -n 1 | cut -c 1-160)"
    fi
  done
  wait "${pid}" || { echo "[fail] evaluation: see $3.log"; exit 1; }
  echo "[done] evaluation in $(( (SECONDS - start) / 60 ))m"
}

setup_env
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"  # this checkout, also if the env was installed from another one
GPUS=${GPUS:-${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)}}
IFS=, read -r -a GPU_LIST <<< "${GPUS}"
EXPERIMENTS=("${@:-llama3.2-3b_28to19}")
[ "${EXPERIMENTS[0]}" = main ] && EXPERIMENTS=("${MAIN[@]}")

NAMES=(); RESULTS=()
for experiment in "${EXPERIMENTS[@]}"; do
  cfg=$(config_path "${experiment}")
  name=$(basename "${cfg}" .yaml)
  work=${OUT_DIR}/${name}
  echo "=== ${name} (GPUs ${GPUS}) -> ${work}"
  prepare_data "${cfg}"

  # ------------------------------------------------------------ 3. layer-wise curriculum + recovery stage
  python scripts/run_curriculum.py --config "${cfg}" --work_dir "${work}" --gpus "${GPUS}" --post_train \
    --jobs_per_gpu "${JOBS_PER_GPU}" --cleanup_features

  if [ "$(config_value "${cfg}" mode)" = online ]; then
    echo "${name}: compressed model in ${work}/post_train (BERT / GPT-2 are evaluated on GLUE / perplexity)"; continue
  fi
  echo "=== ${name}: evaluating (log: ${work}/eval_post_train.log)"
  n=$(config_value "${cfg}" gpus_per_model extract); n=${n:-1}   # GPUs per model copy (2 for 8B and larger)
  evaluate "$(IFS=,; echo "${GPU_LIST[*]:0:n}")" "${work}/stacked" "${work}/eval_post_train" "${work}/post_train"
  python scripts/summarize_results.py "${work}/eval_post_train" --names "${name}" | tee "${work}/results.txt"
  NAMES+=("${name}"); RESULTS+=("${work}/eval_post_train")
done

if [ ${#NAMES[@]} -gt 1 ]; then
  python scripts/summarize_results.py "${RESULTS[@]}" --names "${NAMES[@]}" | tee "${OUT_DIR}/results.txt"
fi
