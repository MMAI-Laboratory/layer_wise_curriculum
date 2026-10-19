"""Layer selections (teacher -> student layer mappings) used in the paper.

A layer selection is a sorted list of teacher layer indices ``S = [s_0, ..., s_{n-1}]``.
The student has ``n`` layers and student layer ``i`` is initialized from teacher layer ``s_i``.
Student layer ``i`` is trained to reproduce the *segment* of teacher layers ``[s_i, s_{i+1} - 1]``:

    input  = hidden state entering teacher layer s_i
    target = hidden state leaving teacher layer s_{i+1} - 1

If ``s_{i+1} == s_i + 1`` the segment is a single layer, so student layer ``i`` is an exact copy
of the teacher layer and is not trained ("inherited"). The last student layer is always inherited.
"""
from typing import Dict, List, Optional

# Layer selections used for the experiments.
LAYER_CONFIGS: Dict[str, List[int]] = {
    # LLaMA-3.2-3B (28 layers)
    "llama3.2-3b_28to19": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 20, 23, 27],
    "llama3.2-3b_28to14": [0, 1, 2, 3, 4, 5, 8, 11, 14, 17, 20, 23, 26, 27], 
    # LLaMA-3.1-8B / LLaMA-2-7B (32 layers)
    "llama-32to23": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 24, 27, 31],  
    "llama-32to16": [0, 1, 2, 3, 4, 5, 6, 7, 10, 13, 16, 19, 22, 25, 28, 31],  
    # LLaMA-2-13B (40 layers)
    "llama-40to30": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                     26, 30, 34, 39],  
    "llama-40to20": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17, 21, 25, 29, 34, 39],  
    # Qwen2.5-7B (28 layers)
    "qwen2.5-7b_28to19": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 20, 23, 27],  
    "qwen2.5-7b_28to14": [0, 1, 2, 3, 4, 5, 6, 9, 12, 15, 18, 21, 24, 27],  
    # Qwen2.5-14B (48 layers)
    "qwen2.5-14b_48to35": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                           28, 29, 30, 31, 32, 33, 34, 38, 42, 47],  
    "qwen2.5-14b_48to24": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 19, 23, 27, 31, 35, 39, 43, 47],  
}


def uniform_selection(num_teacher_layers: int, ratio: int) -> List[int]:
    """Keep every ``ratio``-th layer and always keep the last one (used for BERT / GPT-2)."""
    num_student_layers = int(num_teacher_layers / ratio)
    selection = [i * ratio for i in range(num_student_layers) if i < int(num_student_layers - 1)]
    selection.append(num_teacher_layers - 1)
    return selection


def resolve_layer_config(spec: str, num_teacher_layers: Optional[int] = None) -> List[int]:
    """Resolve a layer selection from

    * a preset name in :data:`LAYER_CONFIGS` (e.g. ``llama3.2-3b_28to19``),
    * ``uniform:<ratio>`` (e.g. ``uniform:2`` keeps every 2nd layer), or
    * an explicit comma-separated list of teacher layer indices (e.g. ``0,1,2,5,8,11``).
    """
    if spec in LAYER_CONFIGS:
        selection = list(LAYER_CONFIGS[spec])
    elif spec.startswith("uniform:"):
        if num_teacher_layers is None:
            raise ValueError("`uniform:<ratio>` requires the number of teacher layers.")
        selection = uniform_selection(num_teacher_layers, int(spec.split(":", 1)[1]))
    elif "," in spec:
        selection = [int(x) for x in spec.split(",")]
    else:
        raise ValueError(f"Unknown layer config '{spec}'. Available presets: {sorted(LAYER_CONFIGS)}")

    if selection != sorted(set(selection)) or selection[0] != 0:
        raise ValueError(f"Layer selection must be strictly increasing and start at 0: {selection}")
    if num_teacher_layers is not None and selection[-1] != num_teacher_layers - 1:
        raise ValueError(f"Layer selection must end at the last teacher layer ({num_teacher_layers - 1}): {selection}")
    return selection


def is_trainable(selection: List[int], student_idx: int) -> bool:
    """Whether student layer ``student_idx`` replaces more than one teacher layer (and thus is trained)."""
    if student_idx >= len(selection) - 1:
        return False
    return selection[student_idx] + 1 != selection[student_idx + 1]


def trainable_layers(selection: List[int]) -> List[int]:
    return [i for i in range(len(selection)) if is_trainable(selection, i)]


def segment_name(selection: List[int], student_idx: int) -> str:
    """Name of the teacher segment replaced by student layer ``student_idx`` (e.g. ``14-16``)."""
    return f"{selection[student_idx]}-{selection[student_idx + 1] - 1}"
