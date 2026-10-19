# The training loop is adapted from Hugging Face `run_clm_no_trainer.py` (Apache License 2.0).
"""Stage 2 (online variant, used for BERT / GPT-2): train one student layer with features computed on the fly
instead of a feature cache. The teacher is kept on the GPU next to the trained layer.

Student layer ``i`` is trained to produce the teacher hidden state leaving teacher layer ``s_{i+1}-1``.

* Phase 1: its input is the teacher hidden state entering teacher layer ``s_i``.
* Phase >= 2 (``--student``): training continues from the stacked student of the previous phase, and the input is
  the hidden state entering student layer ``i`` in that (frozen) student, as with the feature cache of the LLMs.

The trained layer is saved as ``<out_dir>/layer<i>.pt``.
"""
import argparse
import copy
import math
import os
import time

import torch
import wandb
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from transformers import AutoConfig, default_data_collator, get_scheduler

from layer_curriculum.architectures import (build_student_config, get_arch, get_layers, init_student_from_teacher,
                                            load_model, student_from_config)
from layer_curriculum.data import load_tokenized_dataset
from layer_curriculum.layer_selection import is_trainable, resolve_layer_config, segment_name
from layer_curriculum.losses import layer_loss
from layer_curriculum.utils import (freeze_parameters, grouped_parameters, init_accelerator, print_gpu_memory,
                                    resolve_grad_accum, setup_logging)

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Teacher model (HF hub id or path)")
    parser.add_argument("--student", default=None, help="Stacked student of the previous phase (phase >= 2)")
    parser.add_argument("--layer_config", required=True, help="e.g. uniform:2")
    parser.add_argument("--layer_idx", type=int, required=True)
    parser.add_argument("--train_file", required=True, help="Tokenized dataset with input_ids / attention_mask")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--train_batch_size", type=int, default=32)

    group = parser.add_argument_group("optimization")
    group.add_argument("--loss_func", default="cossim", choices=["cossim", "mse"])
    group.add_argument("--lr", type=float, default=1e-4)
    group.add_argument("--lr_scheduler", default="cosine")
    group.add_argument("--warmup_ratio", type=float, default=0.05, help="Linear warm-up, fraction of all steps")
    group.add_argument("--num_warmup_steps", type=int, default=0, help="Overrides --warmup_ratio when > 0")
    group.add_argument("--weight_decay", type=float, default=0.01)
    group.add_argument("--optim", default="AdamW")
    group.add_argument("--beta1", type=float, default=0.9)
    group.add_argument("--beta2", type=float, default=0.999)
    group.add_argument("--grad_accum_steps", type=int, default=1)
    group.add_argument("--global_batch_size", type=int, default=None,
                       help="Sets --grad_accum_steps = global / (train_batch_size * num_processes)")
    group.add_argument("--train_epoch", type=int, default=1, help="Epochs (passes over --train_file) of this phase")

    group = parser.add_argument_group("misc")
    group.add_argument("--seed", type=int, default=7)
    group.add_argument("--use_flash_attn", action="store_true")
    group.add_argument("--trust_remote_code", action="store_true")
    group.add_argument("--benchmark", action="store_true")
    group.add_argument("--log_wandb", action="store_true")
    group.add_argument("--project", default="layer-curriculum")
    group.add_argument("--run_name", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    args.grad_accum_steps = resolve_grad_accum(args.global_batch_size, args.train_batch_size, args.grad_accum_steps)
    # bf16 weights with flash-attention, otherwise fp32 weights with fp16 mixed precision
    accelerator = init_accelerator(args.grad_accum_steps, mixed_precision=None if args.use_flash_attn else "fp16")
    setup_logging(logger, accelerator)
    if args.seed:
        set_seed(args.seed)

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    arch = get_arch(config)
    selection = resolve_layer_config(args.layer_config, getattr(config, arch.num_layers_attr))
    dtype = torch.bfloat16 if args.use_flash_attn else None
    teacher = load_model(args.model, arch, args.use_flash_attn, dtype=dtype, trust_remote_code=args.trust_remote_code)

    train_dataloader = DataLoader(load_tokenized_dataset(args.train_file), shuffle=True,
                                  collate_fn=default_data_collator, batch_size=args.train_batch_size)

    if args.student is None:
        student_config = build_student_config(args.model, selection, args.block_size, args.trust_remote_code)
        student = student_from_config(student_config, arch, args.use_flash_attn, dtype=dtype,
                                      trust_remote_code=args.trust_remote_code)
        student = init_student_from_teacher(teacher, student, arch, selection)
    else:
        student = load_model(args.student, arch, args.use_flash_attn, dtype=dtype, trust_remote_code=args.trust_remote_code)
    teacher.eval()
    layer = get_layers(student, arch)[args.layer_idx]
    if args.student is None:
        student = None
    else:  # phase >= 2: the stacked student stays frozen and provides the input of the trained layer
        layer = copy.deepcopy(layer)
        freeze_parameters(student)
        student.eval()

    if not is_trainable(selection, args.layer_idx):
        logger.info(f"Student layer {args.layer_idx} copies a single teacher layer: nothing to train.")
        return
    if accelerator.is_main_process:
        os.makedirs(args.out_dir, exist_ok=True)
        if args.log_wandb:
            wandb.init(project=args.project, name=f"{args.run_name}_layer{args.layer_idx}", config=vars(args),
                       reinit=True)
    logger.info(f"Selected teacher layers: {selection} | training segment {segment_name(selection, args.layer_idx)}")

    optimizer = getattr(torch.optim, args.optim)(grouped_parameters(layer, args.weight_decay), lr=args.lr,
                                                 betas=(args.beta1, args.beta2))
    # The dataloader is not sharded yet, so these are scheduler steps of all processes together.
    max_train_steps = args.train_epoch * math.ceil(len(train_dataloader) / args.grad_accum_steps)
    num_warmup_steps = args.num_warmup_steps * accelerator.num_processes if args.num_warmup_steps > 0 \
        else math.ceil(max_train_steps * args.warmup_ratio)
    lr_scheduler = get_scheduler(name=args.lr_scheduler, optimizer=optimizer, num_warmup_steps=num_warmup_steps,
                                 num_training_steps=max_train_steps)
    layer, teacher, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
        layer, teacher, optimizer, lr_scheduler, train_dataloader)
    if student is not None:
        student.to(accelerator.device)

    # hidden_states[j] enters layer j: teacher layer s_i (phase 1) or student layer i (phase >= 2)
    input_idx = selection[args.layer_idx] if student is None else args.layer_idx
    target_idx = selection[args.layer_idx + 1]
    for epoch in range(args.train_epoch):
        teacher.eval()
        layer.train()
        train_start = time.time()
        for step, batch in enumerate(train_dataloader):
            with torch.no_grad():
                hidden_states = teacher(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                        output_hidden_states=True).hidden_states
                input_states = hidden_states
                if student is not None:
                    with accelerator.autocast():
                        input_states = student(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                               output_hidden_states=True).hidden_states
            with accelerator.accumulate(layer):
                input_h = input_states[input_idx].clone().detach()
                target_h = hidden_states[target_idx].clone().detach()
                if arch.rotary:
                    position_ids = torch.arange(0, input_h.shape[1]).repeat(input_h.shape[0], 1).to(input_h.device)
                    output_h = layer(input_h, position_ids=position_ids)[0]
                else:
                    output_h = layer(input_h)[0]
                loss = layer_loss(args.loss_func, output_h, target_h)

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if args.benchmark:
                print_gpu_memory()

            if accelerator.is_main_process and step % 500 == 0 and step != 0:
                elapsed = time.time() - train_start
                print(f"--- [{step}/{len(train_dataloader)}] loss {loss.item():.4f}: {int(elapsed // 60)}/"
                      f"{int(elapsed / step * len(train_dataloader) // 60)} min (elapsed/epoch) ---")
                if args.log_wandb:
                    wandb.log({"train/loss": loss.item()})

    layer = accelerator.unwrap_model(layer).cpu()
    accelerator.end_training()
    if accelerator.is_main_process:
        torch.save(layer.state_dict(), os.path.join(args.out_dir, f"layer{args.layer_idx}.pt"))
        if args.log_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
