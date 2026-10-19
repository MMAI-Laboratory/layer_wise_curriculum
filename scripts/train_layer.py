# The training loop is adapted from Hugging Face `run_clm_no_trainer.py` (Apache License 2.0).
"""Stage 2: train one student layer on cached features (one process per layer, layers run in parallel).

Student layer ``i`` (initialized from teacher layer ``s_i``) learns to map the cached input hidden state to
the cached teacher target of its segment ``[s_i, s_{i+1}-1]``. Only this single layer and its features are
on the GPU, so the memory footprint is that of one decoder layer.

* Phase 1: the student is built from the teacher (``--model``) and the layer selection.
* Phase >= 2: pass ``--student <stacked model>`` to continue from the previous phase.

The trained layer is saved as ``<out_dir>/layer<i>.pt``.
"""
import argparse
import math
import os
import time

import torch
import torch.multiprocessing as mp
import wandb
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from transformers import AutoConfig, get_scheduler

from layer_curriculum.architectures import (build_student_config, get_arch, get_layers, init_student_from_teacher,
                                            load_model, student_from_config)
from layer_curriculum.curriculum import update_adaptive_loss_state
from layer_curriculum.feature_cache import (FeatureDataset, QuantFeatureDataset, collate_identity, decode_batch,
                                            feature_files)
from layer_curriculum.layer_selection import is_trainable, resolve_layer_config, segment_name
from layer_curriculum.losses import layer_loss
from layer_curriculum.utils import freeze_parameters, grouped_parameters, init_accelerator, print_gpu_memory, setup_logging

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Teacher model (HF hub id or path)")
    parser.add_argument("--student", default=None, help="Stacked student of the previous phase (phase >= 2)")
    parser.add_argument("--layer_config", required=True)
    parser.add_argument("--layer_idx", type=int, required=True, help="Student layer to train")
    parser.add_argument("--feature_dir", required=True, help="Feature cache written by extract_features.py")
    parser.add_argument("--out_dir", required=True, help="Directory for trained layer weights")
    parser.add_argument("--quant", type=int, default=8, choices=[4, 8, 16])
    parser.add_argument("--block_size", type=int, default=1024)

    group = parser.add_argument_group("optimization")
    group.add_argument("--loss_func", default="cossim", choices=["cossim", "mse", "norm_mse", "kld"])
    group.add_argument("--lr", type=float, default=3e-4)
    group.add_argument("--lr_scheduler", default="linear")
    group.add_argument("--warmup_ratio", type=float, default=0.005, help="Linear warm-up, fraction of all steps")
    group.add_argument("--num_warmup_steps", type=int, default=0, help="Overrides --warmup_ratio when > 0")
    group.add_argument("--weight_decay", type=float, default=1e-3)
    group.add_argument("--optim", default="AdamW", help="Any optimizer in torch.optim")
    group.add_argument("--beta1", type=float, default=0.9)
    group.add_argument("--beta2", type=float, default=0.95)
    group.add_argument("--grad_accum_steps", type=int, default=1024)
    group.add_argument("--train_epoch", type=int, default=20, help="(Maximum) number of epochs of this phase")
    group.add_argument("--max_train_steps", type=int, default=None, help="Overrides --train_epoch")
    group.add_argument("--train_steps", type=int, default=None, help="Cap on micro-steps per epoch (debugging)")

    group = parser.add_argument_group("convergence-aware adaptive schedule")
    group.add_argument("--adaptive", action="store_true", help="Stop once the epoch loss stagnates")
    group.add_argument("--adaptive_loss_delta", type=float, default=1e-4)
    group.add_argument("--adaptive_loss_patience", type=int, default=2)
    group.add_argument("--adaptive_min_epoch", type=int, default=5)

    group = parser.add_argument_group("misc")
    group.add_argument("--seed", type=int, default=7)
    group.add_argument("--use_flash_attn", action="store_true", help="FlashAttention-2 kernels (weights are bf16)")
    group.add_argument("--trust_remote_code", action="store_true")
    group.add_argument("--benchmark", action="store_true", help="Print GPU memory every step")
    group.add_argument("--log_wandb", action="store_true")
    group.add_argument("--project", default="layer-curriculum")
    group.add_argument("--run_name", default="")
    args = parser.parse_args()
    if args.quant == 16:
        args.quant = None
    return args


def build_layer(args, arch, selection):
    """Return the student layer ``args.layer_idx`` (on CPU)."""
    dtype = torch.bfloat16
    if args.student is None:
        teacher = load_model(args.model, arch, args.use_flash_attn, dtype=dtype, trust_remote_code=args.trust_remote_code)
        freeze_parameters(teacher)
        student_config = build_student_config(args.model, selection, args.block_size, args.trust_remote_code)
        student = student_from_config(student_config, arch, args.use_flash_attn, dtype=dtype,
                                      trust_remote_code=args.trust_remote_code)
        student = init_student_from_teacher(teacher, student, arch, selection)
        del teacher
    else:
        student = load_model(args.student, arch, args.use_flash_attn, dtype=dtype, trust_remote_code=args.trust_remote_code)
    return get_layers(student, arch)[args.layer_idx].cpu()


def train(args, accelerator, arch, layer, train_dataloader):
    optimizer = getattr(torch.optim, args.optim)(grouped_parameters(layer, args.weight_decay), lr=args.lr,
                                                 betas=(args.beta1, args.beta2))
    num_samples = len(train_dataloader)
    num_update_steps_per_epoch = math.ceil(num_samples / args.grad_accum_steps)
    overrode_max_train_steps = args.max_train_steps is None
    if overrode_max_train_steps:
        args.max_train_steps = args.train_epoch * num_update_steps_per_epoch
    if args.num_warmup_steps == 0:
        args.num_warmup_steps = math.ceil(args.max_train_steps * args.warmup_ratio)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps if overrode_max_train_steps
        else args.max_train_steps * accelerator.num_processes,
    )
    layer, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(layer, optimizer, lr_scheduler, train_dataloader)
    args.train_epoch = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    logger.info("***** Running layer-wise training *****")
    logger.info(f"  Student layer = {args.layer_idx}")
    logger.info(f"  Trainable parameters = {sum(p.numel() for p in layer.parameters() if p.requires_grad)}")
    logger.info(f"  Cached samples (this process) = {num_samples}")
    logger.info(f"  Num epochs = {args.train_epoch} | adaptive = {args.adaptive}")
    logger.info(f"  Total batch size = {accelerator.num_processes * args.grad_accum_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps} (warm-up {args.num_warmup_steps})")

    dtype = torch.bfloat16
    hidden_size = layer.input_layernorm.weight.shape[0] if hasattr(layer, "input_layernorm") else None
    best_loss, stagnant_epochs = None, 0
    layer.train()
    for epoch in range(args.train_epoch):
        total_loss, num_loss_steps = 0, 0
        train_start = time.time()
        for step, batch in enumerate(train_dataloader):
            input_h, target_h = decode_batch(batch, args.quant, "cuda", dtype, hidden_size)
            with accelerator.accumulate(layer):
                if arch.rotary:
                    position_ids = torch.arange(input_h.shape[1], device="cuda").unsqueeze(0)
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

            total_loss += loss.detach().float()
            num_loss_steps += 1
            if accelerator.is_main_process and step % 500 == 0 and step != 0:
                elapsed = time.time() - train_start
                print(f"--- [{step}/{num_samples}] epoch {epoch}: {int(elapsed // 60)}/"
                      f"{int(elapsed / step * num_samples // 60)} min (elapsed/epoch) ---")
            if args.train_steps is not None and step >= args.train_steps:
                break

        # epoch loss averaged over all processes
        local_sum = total_loss if num_loss_steps > 0 else torch.tensor(0.0, device=accelerator.device)
        stats = torch.stack((local_sum, torch.tensor(float(num_loss_steps), device=local_sum.device))).reshape(1, 2)
        stats = accelerator.gather_for_metrics(stats)
        if stats[:, 1].sum().item() > 0:
            epoch_loss = (stats[:, 0].sum() / stats[:, 1].sum()).item()
            logger.info(f"--- [{epoch + 1}/{args.train_epoch}] epoch - Train loss: {epoch_loss:.6f}")
            if args.log_wandb and accelerator.is_main_process:
                wandb.log({"train/epoch_loss": epoch_loss, "epoch": epoch + 1})
            if args.adaptive:
                best_loss, stagnant_epochs, should_stop = update_adaptive_loss_state(
                    epoch, epoch_loss, best_loss, stagnant_epochs, delta=args.adaptive_loss_delta,
                    patience=args.adaptive_loss_patience, min_epoch=args.adaptive_min_epoch)
                logger.info(f"--- Adaptive loss status: best={best_loss:.6f}, "
                            f"stagnant={stagnant_epochs}/{args.adaptive_loss_patience}")
                if should_stop:
                    logger.info(f"--- Adaptive loss converged at epoch {epoch + 1}. Stop training.")
                    break
    return layer


def main():
    args = parse_args()
    accelerator = init_accelerator(args.grad_accum_steps, mixed_precision=None, iterable_data=True)  # bf16 weights
    setup_logging(logger, accelerator)
    if args.seed:
        set_seed(args.seed)

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    arch = get_arch(config)
    selection = resolve_layer_config(args.layer_config, getattr(config, arch.num_layers_attr))
    if not is_trainable(selection, args.layer_idx):
        logger.info(f"Student layer {args.layer_idx} copies a single teacher layer: nothing to train.")
        return

    if accelerator.is_main_process:
        os.makedirs(args.out_dir, exist_ok=True)
        if args.log_wandb:
            wandb.init(project=args.project, name=f"{args.run_name}_layer{args.layer_idx}", config=vars(args),
                       reinit=True)

    layer = build_layer(args, arch, selection)
    logger.info(f"Selected teacher layers: {selection} | training segment {segment_name(selection, args.layer_idx)}")

    feature_dir = os.path.join(args.feature_dir, segment_name(selection, args.layer_idx))
    files = feature_files(feature_dir, args.quant)
    if not files:
        raise FileNotFoundError(f"No cached features found in {feature_dir}")
    if args.quant is not None:
        dataset = QuantFeatureDataset(files, process_index=accelerator.process_index,
                                      num_processes=accelerator.num_processes)
    else:
        dataset = FeatureDataset(files, dtype=torch.bfloat16)
    train_dataloader = DataLoader(dataset, batch_size=1, num_workers=1, persistent_workers=True, pin_memory=False,
                                  prefetch_factor=4, multiprocessing_context=mp.get_context("spawn"),
                                  collate_fn=collate_identity if args.quant is not None else None)

    layer = train(args, accelerator, arch, layer, train_dataloader)
    layer = accelerator.unwrap_model(layer).cpu()
    accelerator.wait_for_everyone()
    accelerator.end_training()
    if accelerator.is_main_process:
        torch.save(layer.state_dict(), os.path.join(args.out_dir, f"layer{args.layer_idx}.pt"))
    if args.log_wandb and accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
