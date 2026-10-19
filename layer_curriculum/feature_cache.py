"""On-disk cache of teacher (or student) hidden states used as layer-wise inputs / targets.

Hidden states are stored with outlier-aware absmax quantization: feature dimensions that contain an
activation with ``|x| >= alpha`` are kept in full precision and every other dimension is quantized with
a single per-tensor absmax scale. Each cached sample is one ``.pt`` file holding an (input, target) pair.

* ``quant=8``:    ``(input_dict, target_dict)`` with keys ``q`` (int8), ``scale``, ``outlier_vals``, ``outlier_dims``
* ``quant=4``:    a single dict; input and target 4-bit codes are packed into one uint8 tensor
* ``quant=None``: fp16 ``safetensors`` files holding batches of 100 (input, target) pairs
"""
import glob
import math
import os

import torch
from safetensors.torch import load_file
from torch.utils.data import IterableDataset


# ----------------------------------------------------------------------------- quantization

def find_outlier_dims(x: torch.Tensor, alpha: float = 6.0) -> torch.Tensor:
    outlier_mask = (x.abs() >= alpha).any(dim=0)
    return torch.where(outlier_mask)[0]


def absmax_quantize_exclude_outlier(x: torch.Tensor, outlier_dims: torch.Tensor, bits: int, device="cpu"):
    outlier_vals = x[:, outlier_dims]
    non_outlier_vals = x.clone()
    non_outlier_vals[:, outlier_dims] = 0

    scale_factor = 2 ** (bits - 1) - 1
    scale = scale_factor / torch.max(torch.abs(non_outlier_vals))
    q_x = (scale * non_outlier_vals).round().to(torch.int8)
    return {
        "q": q_x,
        "scale": torch.tensor([scale], device=device),
        "outlier_vals": outlier_vals,
        "outlier_dims": outlier_dims.to(torch.int16),
    }


def quantize_hidden_state(h: torch.Tensor, bits: int, alpha: float = 6.0, device="cpu"):
    """``h``: [1, seq_len, hidden] float32 hidden state."""
    h = h.squeeze(0)
    outlier_dims = find_outlier_dims(h, alpha=alpha)
    return absmax_quantize_exclude_outlier(h, outlier_dims, bits, device=device)


def pack_int4_pair(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    q1 = q1.clamp_(-7, 7).to(torch.int16) + 7
    q2 = q2.clamp_(-7, 7).to(torch.int16) + 7
    return ((q1.to(torch.uint8) << 4) | q2.to(torch.uint8)).to(torch.uint8)


def make_quant_pair(input_h: torch.Tensor, target_h: torch.Tensor, bits: int):
    input_q = quantize_hidden_state(input_h, bits)
    target_q = quantize_hidden_state(target_h, bits)
    if bits == 8:
        return input_q, target_q
    if bits == 4:
        return {
            "packed": pack_int4_pair(input_q["q"], target_q["q"]),
            "input_scale": input_q["scale"],
            "input_outlier_vals": input_q["outlier_vals"],
            "input_outlier_dims": input_q["outlier_dims"],
            "label_scale": target_q["scale"],
            "label_outlier_vals": target_q["outlier_vals"],
            "label_outlier_dims": target_q["outlier_dims"],
        }
    raise ValueError(f"Unsupported number of bits: {bits}")


def _restore_outliers(x, dims, vals, device, dtype):
    if dims is not None and dims.numel() > 0:
        idx = dims.to(device=device, dtype=torch.long)
        x.index_copy_(1, idx, vals.to(device=device, dtype=dtype))
    return x


def dequantize_int8(quant: dict, device, dtype, expected_hidden_size=None) -> torch.Tensor:
    """Returns a [1, seq_len, hidden] tensor."""
    q = quant["q"].to(device=device, dtype=dtype, non_blocking=True)
    scale = quant["scale"].to(device=device, dtype=dtype, non_blocking=True)
    if scale.dim() == 0:
        scale = scale.view(1)
    x = _restore_outliers(q / scale, quant.get("outlier_dims"), quant.get("outlier_vals"), device, dtype)
    if expected_hidden_size is not None:
        assert x.size(-1) == expected_hidden_size, f"hidden size mismatch: {x.size(-1)} vs {expected_hidden_size}"
    return x.unsqueeze(0).contiguous()


def dequantize_int4_pair(quant: dict, device, dtype, expected_hidden_size=None):
    """Returns ([1, seq_len, hidden] input, [1, seq_len, hidden] target)."""
    packed = quant["packed"].to(device=device, dtype=torch.uint8, non_blocking=True)
    codes = {
        "input": ((packed >> 4) & 0x0F).to(torch.int16) - 7,
        "label": (packed & 0x0F).to(torch.int16) - 7,
    }
    out = []
    for prefix in ("input", "label"):
        scale = quant[f"{prefix}_scale"].to(device=device, dtype=dtype, non_blocking=True)
        if scale.dim() == 0:
            scale = scale.view(1)
        x = codes[prefix].to(dtype=dtype) / scale
        x = _restore_outliers(x, quant.get(f"{prefix}_outlier_dims"), quant.get(f"{prefix}_outlier_vals"), device, dtype)
        if expected_hidden_size is not None:
            assert x.size(-1) == expected_hidden_size, f"hidden size mismatch: {x.size(-1)} vs {expected_hidden_size}"
        out.append(x.unsqueeze(0).contiguous())
    return tuple(out)


def decode_batch(batch, quant, device, dtype, expected_hidden_size=None):
    """Turn one item from a feature DataLoader into (input, target) hidden states on ``device``."""
    if quant == 8:
        input_q, target_q = batch
        return (dequantize_int8(input_q, device, dtype, expected_hidden_size),
                dequantize_int8(target_q, device, dtype, expected_hidden_size))
    if quant == 4:
        return dequantize_int4_pair(batch, device, dtype, expected_hidden_size)
    return batch[0].squeeze(0).to(device=device), batch[1].squeeze(0).to(device=device)


# ----------------------------------------------------------------------------- datasets

def feature_files(feature_dir: str, quant):
    ext = "pt" if quant is not None else "safetensors"
    return sorted(glob.glob(os.path.join(feature_dir, f"*.{ext}")))


class QuantFeatureDataset(IterableDataset):
    """Streams quantized (input, target) pairs, one ``.pt`` file per sample.
    Files are sharded across processes (``process_index``) and DataLoader workers."""

    def __init__(self, file_paths, process_index=0, num_processes=1):
        self.file_paths = list(file_paths)[process_index::num_processes]

    def __len__(self):
        return len(self.file_paths)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        files = self.file_paths
        if worker_info is not None:
            per_worker = math.ceil(len(files) / worker_info.num_workers)
            start = worker_info.id * per_worker
            files = files[start:min(start + per_worker, len(files))]
        for path in files:
            yield torch.load(path, map_location="cpu")


class FeatureDataset(IterableDataset):
    """Streams fp16 (input, target) pairs stored in ``safetensors`` files (``quant=None``)."""

    def __init__(self, file_paths, dtype=torch.float16, batch_size=1):
        self.file_paths = list(file_paths)
        self.dtype = dtype
        self.batch_size = batch_size

    def __len__(self):
        return sum(load_file(path)["input"].shape[0] for path in self.file_paths)

    def __iter__(self):
        input_batch, target_batch = [], []
        for path in self.file_paths:
            data = load_file(path)
            for input_, target in zip(data["input"].to(self.dtype), data["label"].to(self.dtype)):
                input_batch.append(input_)
                target_batch.append(target)
                if len(input_batch) == self.batch_size:
                    yield torch.stack(input_batch), torch.stack(target_batch)
                    input_batch, target_batch = [], []
        if input_batch:
            yield torch.stack(input_batch), torch.stack(target_batch)


def collate_identity(batch):
    return batch[0]
