import torch
import torch.nn.functional as F

LOSSES = ("cossim", "mse", "norm_mse", "kld")


def layer_loss(loss_func: str, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Local (layer-wise) distillation loss between student layer output and teacher target hidden states."""
    if loss_func in ("cossim", "cos_sim"):
        return 1.0 - F.cosine_similarity(output, target, dim=-1).mean()
    if loss_func == "mse":
        return F.mse_loss(output, target)
    if loss_func == "norm_mse":
        return F.mse_loss(output, target) / (torch.mean(target ** 2) + 1e-8)
    if loss_func == "kld":
        return F.kl_div(F.log_softmax(output, dim=-1), F.softmax(target, dim=-1))
    raise ValueError(f"Unknown loss function '{loss_func}' (choose from {LOSSES})")
