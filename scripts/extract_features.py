"""Stage 1: cache the (input, target) hidden states of every trainable student layer.

For student layer ``i`` replacing teacher segment ``[s_i, s_{i+1}-1]`` we store

    input  = teacher hidden state entering layer s_i            (phase 1, no --student)
           = student hidden state entering student layer i      (phase >= 2, --student <stacked model>)
    target = teacher hidden state leaving layer s_{i+1} - 1

Each GPU group (``--gpus_per_model`` GPUs) holds one copy of the model(s) and processes a round-robin share
of the batches. Every sample is kept for a segment with probability 0.5 so that segments see different
subsets of the corpus, until ``--num_sample`` samples per segment are stored.

Output layout: ``<out_dir>/<segment>/quant_dict_<id>.pt`` (``--quant 8|4``) or
``<out_dir>/<segment>/feature_pair_<id>.safetensors`` (no quantization).
"""
import argparse
import math
import os
import random
from collections import defaultdict

import torch
import torch.multiprocessing as mp
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from transformers import AutoConfig, default_data_collator

from layer_curriculum.architectures import get_arch, load_model
from layer_curriculum.data import load_tokenized_dataset
from layer_curriculum.feature_cache import make_quant_pair
from layer_curriculum.layer_selection import resolve_layer_config, segment_name


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Teacher model (HF hub id or path)")
    parser.add_argument("--student", default=None, help="Stacked student of the previous phase (phase >= 2)")
    parser.add_argument("--layer_config", required=True, help="Preset name, 'uniform:<r>' or comma-separated list")
    parser.add_argument("--train_file", required=True, help="Tokenized dataset (see scripts/prepare_slimpajama.py)")
    parser.add_argument("--holdout_percentage", type=int, default=5,
                        help="The first N%% of the blocks are reserved for validation and not used")
    parser.add_argument("--out_dir", required=True, help="Directory to store the cached features")
    parser.add_argument("--num_sample", type=int, default=15000, help="Number of cached samples per segment")
    parser.add_argument("--active_layers", default=None,
                        help="Comma-separated student layers whose segments are written (default: all). The other "
                             "segments are sampled but not stored, so the stored samples do not depend on this option")
    parser.add_argument("--quant", type=int, default=8, choices=[4, 8, 16],
                        help="Bits used to store the features (16: fp16 safetensors, no quantization)")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gpus_per_model", type=int, default=1,
                        help="GPUs used to shard one model copy (e.g. 1 for 3B, 2 for 7/8B on 24GB GPUs)")
    parser.add_argument("--num_threads", type=int, default=None,
                        help="CPU threads per model copy (default: CPU cores / number of model copies)")
    parser.add_argument("--use_flash_attn", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()
    if args.quant == 16:
        args.quant = None
    return args


def _device_map_and_memory(gpus_per_model, assigned_gpus, total_visible_gpus):
    if gpus_per_model == 1:
        return {"": assigned_gpus[0]}, None
    max_memory = {}
    for gpu_id in range(total_visible_gpus):
        if gpu_id in assigned_gpus:
            # Reserve ~15% headroom for activations
            total = torch.cuda.get_device_properties(gpu_id).total_memory
            max_memory[gpu_id] = f"{int(total * 0.85 / (1024 ** 3))}GiB"
        else:
            max_memory[gpu_id] = "0GiB"
    return "auto", max_memory


def _to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    return type(obj)(_to_cpu(v) for v in obj)


def worker(group_rank, num_groups, args):
    # the model copies run in parallel processes: share the CPU cores between them
    torch.set_num_threads(args.num_threads or max(1, (os.cpu_count() or 1) // num_groups))
    total_visible_gpus = torch.cuda.device_count()
    assigned_gpus = list(range(group_rank * args.gpus_per_model, (group_rank + 1) * args.gpus_per_model))
    torch.cuda.set_device(assigned_gpus[0])
    random.seed(42 + group_rank)  # segment-wise random sub-sampling, different per group
    print(f"[Group {group_rank}] Assigned GPUs: {assigned_gpus}")

    device_map, max_memory = _device_map_and_memory(args.gpus_per_model, assigned_gpus, total_visible_gpus)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    arch = get_arch(config)
    dtype = torch.bfloat16
    load_kwargs = dict(device_map=device_map, max_memory=max_memory, use_cache=False)

    teacher = load_model(args.model, arch, args.use_flash_attn, dtype=dtype,
                         trust_remote_code=args.trust_remote_code, **load_kwargs).eval()
    teacher_device = next(iter(teacher.parameters())).device
    if args.student is not None:
        student = load_model(args.student, arch, args.use_flash_attn, dtype=dtype,
                             trust_remote_code=args.trust_remote_code, **load_kwargs).eval()
        student_device = next(iter(student.parameters())).device

    dataloader = DataLoader(load_tokenized_dataset(args.train_file, args.holdout_percentage), collate_fn=default_data_collator,
                            batch_size=args.batch_size)
    selection = resolve_layer_config(args.layer_config, config.num_hidden_layers)

    samples_per_group = math.ceil(args.num_sample / num_groups)
    sample_offset = group_rank * samples_per_group

    # None: segment is a single layer (nothing to cache) or already finished
    sample_count = [None if selection[i + 1] - 1 == selection[i] else 0 for i in range(len(selection) - 1)]
    active = None if args.active_layers is None else {int(i) for i in args.active_layers.split(",")}
    fp16_cache = defaultdict(list)
    for step, batch in enumerate(dataloader):
        if step % num_groups != group_rank:  # round-robin over model groups
            continue

        with torch.no_grad():
            teacher_hidden = teacher(**{k: v.to(teacher_device) for k, v in batch.items()},
                                     output_hidden_states=True).hidden_states
            if args.student is not None:
                student_hidden = student(**{k: v.to(student_device) for k, v in batch.items()},
                                         output_hidden_states=True).hidden_states

        for i in range(len(selection) - 1):
            if sample_count[i] is None:
                continue
            if random.random() >= 0.5:  # keep every sample with probability 0.5
                continue
            if active is not None and i not in active:  # frozen layer: advance the sampling state only
                sample_count[i] += 1
                if sample_count[i] >= samples_per_group:
                    sample_count[i] = None
                continue
            segment = segment_name(selection, i)
            input_h = (teacher_hidden[selection[i]] if args.student is None else student_hidden[i]).to(torch.float32)
            target_h = teacher_hidden[selection[i + 1]].to(torch.float32).to(input_h.device)
            save_dir = os.path.join(args.out_dir, segment)
            os.makedirs(save_dir, exist_ok=True)
            sample_count[i] += 1
            count = sample_count[i]
            sample_id = sample_offset + count

            if args.quant is not None:
                if count <= samples_per_group:  # quantized on the GPU, stored on the CPU
                    torch.save(_to_cpu(make_quant_pair(input_h, target_h, args.quant)),
                               os.path.join(save_dir, f"quant_dict_{sample_id}.pt"))
                if count % 100 == 0 or count >= samples_per_group:
                    print(f"[Group {group_rank}] Saved {count}/{samples_per_group} feature pairs of segment {segment}")
            else:
                fp16_cache[segment].append((input_h.cpu(), target_h.cpu()))
                if count % 100 == 0 or count >= samples_per_group:
                    save_file({"input": torch.stack([x[0] for x in fp16_cache[segment]]).half(),
                               "label": torch.stack([x[1] for x in fp16_cache[segment]]).half()},
                              os.path.join(save_dir, f"feature_pair_{sample_id}.safetensors"))
                    print(f"[Group {group_rank}] Saved {count}/{samples_per_group} feature pairs of segment {segment}")
                    fp16_cache[segment] = []

            if count >= samples_per_group:
                sample_count[i] = None

        if step >= args.num_sample and all(c is None for c in sample_count):
            break

    print(f"[Group {group_rank}] Feature extraction finished.")


def main():
    args = parse_args()
    total_gpus = torch.cuda.device_count()
    assert total_gpus >= args.gpus_per_model and total_gpus % args.gpus_per_model == 0, \
        f"{total_gpus} visible GPU(s) cannot be split into groups of {args.gpus_per_model}"
    num_groups = total_gpus // args.gpus_per_model
    print(f"{num_groups} model group(s), {args.gpus_per_model} GPU(s) per model")
    if num_groups > 1:
        mp.spawn(worker, args=(num_groups, args), nprocs=num_groups, join=True)
    else:
        worker(0, 1, args)


if __name__ == "__main__":
    main()
