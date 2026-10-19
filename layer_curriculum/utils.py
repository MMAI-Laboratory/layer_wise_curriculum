import logging
from datetime import timedelta

import datasets
import torch
import transformers
from accelerate import Accelerator, PartialState
from accelerate.utils import InitProcessGroupKwargs


def init_accelerator(grad_accum_steps: int = 1, mixed_precision=None, iterable_data: bool = False):
    return Accelerator(mixed_precision=mixed_precision, gradient_accumulation_steps=grad_accum_steps,
                       kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=1800000))],
                       dispatch_batches=False if iterable_data else None)


def resolve_grad_accum(global_batch_size, per_device_batch_size: int, grad_accum_steps: int) -> int:
    """Gradient accumulation steps that give ``global_batch_size`` on the launched number of processes."""
    if global_batch_size is None:
        return grad_accum_steps
    num_processes = PartialState().num_processes
    per_step = per_device_batch_size * num_processes
    if global_batch_size % per_step != 0:
        raise ValueError(f"global batch size {global_batch_size} is not divisible by per-device batch size "
                         f"{per_device_batch_size} x {num_processes} process(es)")
    return global_batch_size // per_step


def setup_logging(logger, accelerator):
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()


def freeze_parameters(module):
    for param in module.parameters():
        param.requires_grad = False


def grouped_parameters(module, weight_decay: float, no_decay=("bias", "layer_norm.weight")):
    """AdamW parameter groups; parameters whose name contains an entry of ``no_decay`` get no weight decay."""
    return [
        {"params": [p for n, p in module.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": weight_decay},
        {"params": [p for n, p in module.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]


def print_gpu_memory():
    print(f"GPU Memory - Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
