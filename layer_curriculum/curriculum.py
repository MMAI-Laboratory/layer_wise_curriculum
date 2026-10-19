"""Layer-wise curriculum schedule.

Let ``T = [t_1, ..., t_K]`` be the trainable student layers (shallow -> deep, see ``layer_selection``).
The curriculum runs ``K`` phases. In phase ``p`` the active layers are ``T[p-1:]``: the shallowest active
layer of the previous phase is frozen and the remaining (deeper) layers keep training.

* Phase 1 trains every trainable layer independently on cached *teacher* features (input and target).
* Phase p >= 2 re-extracts features with the student stacked at the end of phase p-1: the input of each
  active layer is the output of the (partially trained) student below it, the target stays the teacher.
  This aligns the input distribution of deeper layers with what they see at inference time.

How long each phase lasts follows the static exponential schedule ``psi(t) = exp(alpha * t) + beta`` of Appendix C
(:func:`schedule_epochs`, ``alpha = 0.035``). With ``adaptive: true`` the epochs of a phase are an upper bound and a
layer stops as soon as its epoch loss stagnates (:func:`update_adaptive_loss_state`).
"""
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .layer_selection import trainable_layers


@dataclass
class Phase:
    index: int                 
    active_layers: List[int]   
    epochs: int
    from_student: bool         


def schedule_epochs(num_phases: int, alpha: float, beta: float = 0.0) -> List[int]:
    """Epochs of each phase for the exponential schedule ``psi(t) = exp(alpha * t) + beta`` (Appendix C).

    ``psi(t)`` is the index of the shallowest active layer, evaluated at integer epochs ``t`` and discretized as
    ``floor(psi(t))`` (``beta = 0``: ``psi(0) = 1``). Phase ``p`` (1-based) ends at the first epoch ``t >= t_p`` with
    ``psi(t_p) = p + 1``, i.e. ``t_p = log(p + 1 - beta) / alpha``. For example ``alpha = 0.035`` gives
    ``[20, 12, 8, 6]`` for 4 phases: the exponential schedule gives more epochs to the early phases.
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    epochs, start = [], 0
    for p in range(1, num_phases + 1):
        if p + 1 - beta <= 0:
            raise ValueError(f"beta = {beta} is too large for the exponential schedule")
        end = max(math.ceil(math.log(p + 1 - beta) / alpha - 1e-9), start + 1)  # at least one epoch per phase
        epochs.append(end - start)
        start = end
    return epochs


def build_phases(selection: List[int], alpha: float = 0.035, beta: float = 0.0) -> List[Phase]:
    """Curriculum phases for a layer selection, with the epochs of the exponential schedule (:func:`schedule_epochs`)."""
    trainable = trainable_layers(selection)
    epochs = schedule_epochs(len(trainable), alpha, beta)
    return [Phase(index=p + 1, active_layers=trainable[p:], epochs=epochs[p], from_student=p > 0)
            for p in range(len(trainable))]


def update_adaptive_loss_state(epoch: int, epoch_loss: float, best_loss: Optional[float], stagnant_epochs: int,
                               delta: float = 1e-4, patience: int = 2, min_epoch: int = 5
                               ) -> Tuple[float, int, bool]:
    """Convergence-aware early stopping of a single layer.

    The reference loss is refreshed only if the epoch loss improves by more than ``delta``; otherwise the
    stagnation counter grows. The layer is considered converged once it stagnated for ``patience`` epochs
    and at least ``min_epoch`` epochs were run. Returns ``(best_loss, stagnant_epochs, should_stop)``.
    """
    if best_loss is None:
        return epoch_loss, 0, False
    if best_loss - epoch_loss > delta:
        return epoch_loss, 0, False
    stagnant_epochs += 1
    should_stop = (epoch + 1) >= min_epoch and stagnant_epochs >= patience
    return best_loss, stagnant_epochs, should_stop
