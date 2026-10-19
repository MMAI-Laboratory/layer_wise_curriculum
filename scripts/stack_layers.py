"""Stage 3: assemble the trained layers into a compressed model saved in standard Hugging Face format.

* trained layers are loaded from ``<layer_dir>/layer<i>.pt``,
* layers that copy a single teacher layer (and the last layer) are taken from the teacher,
* embeddings, the final norm and the output head are copied from the teacher.

The output can be loaded with ``AutoModelForCausalLM.from_pretrained`` and evaluated with lm-eval directly.
"""
import argparse
import os

import torch
from transformers import AutoConfig, AutoTokenizer

from layer_curriculum.architectures import (build_student_config, get_arch, get_layers, get_submodule, load_model,
                                            student_from_config)
from layer_curriculum.layer_selection import is_trainable, resolve_layer_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Teacher model (HF hub id or path)")
    parser.add_argument("--layer_config", required=True)
    parser.add_argument("--layer_dir", required=True, help="Directory with layer<i>.pt files from train_layer.py")
    parser.add_argument("--save_dir", required=True, help="Output directory of the stacked student")
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--use_flash_attn", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def stack_layers(model, layer_config, layer_dir, block_size=1024, use_flash_attn=False, trust_remote_code=False):
    config = AutoConfig.from_pretrained(model, trust_remote_code=trust_remote_code)
    arch = get_arch(config)
    selection = resolve_layer_config(layer_config, getattr(config, arch.num_layers_attr))
    dtype = torch.bfloat16 if use_flash_attn else None

    teacher = load_model(model, arch, use_flash_attn, dtype=dtype, trust_remote_code=trust_remote_code)
    student_config = build_student_config(model, selection, block_size, trust_remote_code)
    student = student_from_config(student_config, arch, use_flash_attn, dtype=dtype, trust_remote_code=trust_remote_code)

    teacher_layers, student_layers = get_layers(teacher, arch), get_layers(student, arch)
    for idx in range(len(selection)):
        path = os.path.join(layer_dir, f"layer{idx}.pt")
        if os.path.exists(path):
            print(f"Layer {idx}: loading {path}")
            student_layers[idx].load_state_dict(torch.load(path, map_location="cpu"))
        elif not is_trainable(selection, idx):
            print(f"Layer {idx}: copying teacher layer {selection[idx]}")
            student_layers[idx].load_state_dict(teacher_layers[selection[idx]].state_dict())
        else:
            raise FileNotFoundError(f"Trained weights of student layer {idx} not found: {path}")

    # Embeddings, final norm and output head are inherited from the teacher.
    teacher_base, student_base = getattr(teacher, arch.base), getattr(student, arch.base)
    for name in arch.embeddings:
        get_submodule(student_base, name).load_state_dict(get_submodule(teacher_base, name).state_dict())
    if arch.final_norm:
        get_submodule(student_base, arch.final_norm).load_state_dict(get_submodule(teacher_base, arch.final_norm).state_dict())
    getattr(student, arch.head).load_state_dict(getattr(teacher, arch.head).state_dict())
    return student


def main():
    args = parse_args()
    student = stack_layers(args.model, args.layer_config, args.layer_dir, args.block_size, args.use_flash_attn,
                           args.trust_remote_code)
    student.save_pretrained(args.save_dir)
    AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code).save_pretrained(args.save_dir)
    print(f"Saved stacked student to {args.save_dir}")


if __name__ == "__main__":
    main()
