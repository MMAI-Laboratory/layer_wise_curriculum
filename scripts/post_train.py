# The training loop is adapted from Hugging Face `run_clm_no_trainer.py` (Apache License 2.0).
"""Optional stage 4: short end-to-end language-model training of the stacked student ("healing").

* LLMs:    causal LM loss with LoRA (``--lora``) on SlimPajama, as reported for "w/ post-training" results.
* BERT:    ``--mlm`` full fine-tuning with the masked LM objective.
* GPT-2:   causal LM full fine-tuning.

With ``--lora`` only the adapter is saved; evaluate it with
``lm_eval --model hf --model_args pretrained=<stacked>,peft=<adapter>``.
"""
import argparse
import math
import os
import time

import torch
import wandb
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import (AutoConfig, AutoTokenizer, DataCollatorForLanguageModeling, default_data_collator,
                          get_scheduler)

from layer_curriculum.architectures import get_arch, load_model
from layer_curriculum.data import load_tokenized_dataset
from layer_curriculum.utils import grouped_parameters, init_accelerator, resolve_grad_accum, setup_logging

logger = get_logger(__name__)

# Default LoRA targets. For LLMs: the attention projections, as for the adapters behind the paper results.
LORA_TARGETS = {
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "qwen2": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "gpt2": ["c_attn", "c_proj", "c_fc"],
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Stacked student (output of stack_layers.py)")
    parser.add_argument("--train_file", required=True, help="Tokenized dataset")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--mlm", action="store_true", help="Masked LM objective (BERT)")
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--global_batch_size", type=int, default=None,
                        help="Sets --grad_accum_steps = global / (train_batch_size * num_processes)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", default="linear")
    parser.add_argument("--warmup_ratio", type=float, default=0.005)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--optim", default="AdamW")
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--train_epoch", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_target_modules", default=None,
                        help="Comma-separated module names (default: attention projections, see LORA_TARGETS)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--use_flash_attn", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--log_wandb", action="store_true")
    parser.add_argument("--project", default="layer-curriculum-post-train")
    parser.add_argument("--run_name", default="post_train")
    return parser.parse_args()


def main():
    args = parse_args()
    args.grad_accum_steps = resolve_grad_accum(args.global_batch_size, args.train_batch_size, args.grad_accum_steps)
    accelerator = init_accelerator(args.grad_accum_steps, mixed_precision=None if args.use_flash_attn else "fp16")
    setup_logging(logger, accelerator)
    if args.seed:
        set_seed(args.seed)

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    arch = get_arch(config)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = load_model(args.model, arch, args.use_flash_attn, dtype=torch.bfloat16 if args.use_flash_attn else None,
                       trust_remote_code=args.trust_remote_code)
    if args.lora:
        targets = args.lora_target_modules.split(",") if args.lora_target_modules else LORA_TARGETS[config.model_type]
        model = get_peft_model(model, LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha, target_modules=targets,
                                                 lora_dropout=args.lora_dropout, bias="none", task_type="CAUSAL_LM"))
    if args.log_wandb and accelerator.is_main_process:
        wandb.init(project=args.project, name=args.run_name, config=vars(args), reinit=True)

    train_dataset = load_tokenized_dataset(args.train_file)
    if args.mlm:
        collate_fn = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    else:
        if "labels" not in train_dataset.column_names:
            train_dataset = train_dataset.map(lambda example: {"labels": example["input_ids"]}, keep_in_memory=True)
        collate_fn = default_data_collator
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=args.train_batch_size, shuffle=True)

    no_decay = ("bias", "LayerNorm.weight") if args.mlm else ("bias", "layer_norm.weight")
    optimizer = getattr(torch.optim, args.optim)(grouped_parameters(model, args.weight_decay, no_decay), lr=args.lr,
                                                 betas=(args.beta1, args.beta2), eps=args.adam_eps)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.grad_accum_steps)
    overrode_max_train_steps = args.max_train_steps is None
    if overrode_max_train_steps:
        args.max_train_steps = args.train_epoch * num_update_steps_per_epoch
    num_training_steps = args.max_train_steps if overrode_max_train_steps \
        else args.max_train_steps * accelerator.num_processes
    num_warmup_steps = math.ceil(num_training_steps * args.warmup_ratio)  # both in scheduler steps
    lr_scheduler = get_scheduler(name=args.lr_scheduler, optimizer=optimizer, num_warmup_steps=num_warmup_steps,
                                 num_training_steps=num_training_steps)
    model, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(model, optimizer, lr_scheduler,
                                                                           train_dataloader)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.grad_accum_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.train_epoch * num_update_steps_per_epoch
    args.train_epoch = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    logger.info("***** Running post-training *****")
    logger.info(f"  Trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    logger.info(f"  Num examples = {len(train_dataset)} | epochs = {args.train_epoch}")
    logger.info(f"  Total batch size = {args.train_batch_size * accelerator.num_processes * args.grad_accum_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    completed_steps = 0
    for epoch in range(args.train_epoch):
        model.train()
        log_loss, log_steps, train_start = 0, 0, time.time()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                loss = model(**batch).loss
                log_loss += loss.detach().float()
                log_steps += 1
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                completed_steps += 1
            if accelerator.is_main_process and step % 500 == 0 and step != 0:
                avg_loss = (log_loss / log_steps).item()
                elapsed = time.time() - train_start
                print(f"[epoch {epoch} | step {step}/{len(train_dataloader)}] loss={avg_loss:.4f} "
                      f"lr={lr_scheduler.get_last_lr()[0]:.2e} ({int(elapsed // 60)}/"
                      f"{int(elapsed / step * len(train_dataloader) // 60)} min)")
                if args.log_wandb:
                    wandb.log({"train_loss": avg_loss, "learning_rate": lr_scheduler.get_last_lr()[0]},
                              step=completed_steps)
                log_loss, log_steps = 0, 0
            if completed_steps >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    save_dir = os.path.join(args.out_dir, args.run_name)
    accelerator.unwrap_model(model).save_pretrained(save_dir, is_main_process=accelerator.is_main_process,
                                                    save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(save_dir)
        print(f"Saved to {save_dir}")
    if args.log_wandb and accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    main()
