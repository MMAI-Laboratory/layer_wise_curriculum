"""Architecture-specific module names.

To support a new decoder-only family, add an :class:`ArchSpec` entry keyed by ``config.model_type``.
"""
from dataclasses import dataclass
from typing import Tuple

from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMaskedLM


@dataclass(frozen=True)
class ArchSpec:
    base: str                     # attribute holding the backbone (e.g. ``model``)
    layers: str                   # dotted path (relative to ``base``) of the layer ModuleList
    embeddings: Tuple[str, ...]   # dotted paths (relative to ``base``) of embedding modules copied from the teacher
    final_norm: str               # attribute (relative to ``base``) of the final norm, "" if none
    head: str                     # attribute of the output head
    num_layers_attr: str          # config attribute holding the number of layers
    rotary: bool                  # layers take ``position_ids`` (RoPE)
    masked_lm: bool = False

    @property
    def layer_prefix(self) -> str:
        return f"{self.base}.{self.layers}."


_LLAMA_LIKE = ArchSpec(base="model", layers="layers", embeddings=("embed_tokens",), final_norm="norm",
                       head="lm_head", num_layers_attr="num_hidden_layers", rotary=True)

ARCHITECTURES = {
    "llama": _LLAMA_LIKE,      # LLaMA-2 / LLaMA-3.x
    "qwen2": _LLAMA_LIKE,      # Qwen2.5
    "gpt2": ArchSpec(base="transformer", layers="h", embeddings=("wte", "wpe"), final_norm="ln_f",
                     head="lm_head", num_layers_attr="n_layer", rotary=False),
    "bert": ArchSpec(base="bert", layers="encoder.layer",
                     embeddings=("embeddings.word_embeddings", "embeddings.position_embeddings",
                                 "embeddings.token_type_embeddings", "embeddings.LayerNorm"),
                     final_norm="", head="cls", num_layers_attr="num_hidden_layers", rotary=False, masked_lm=True),
}


def get_arch(config) -> ArchSpec:
    if config.model_type not in ARCHITECTURES:
        raise ValueError(f"Unsupported model_type '{config.model_type}'. "
                         f"Add an entry to layer_curriculum/architectures.py (supported: {sorted(ARCHITECTURES)}).")
    return ARCHITECTURES[config.model_type]


def get_submodule(module, dotted: str):
    for name in dotted.split("."):
        module = getattr(module, name)
    return module


def get_layers(model, arch: ArchSpec):
    return get_submodule(getattr(model, arch.base), arch.layers)


def load_model(path, arch: ArchSpec, use_flash_attn: bool, dtype=None, config=None, trust_remote_code=False, **kwargs):
    if arch.masked_lm:
        return AutoModelForMaskedLM.from_pretrained(path, config=config, trust_remote_code=trust_remote_code, **kwargs)
    return AutoModelForCausalLM.from_pretrained(
        path, config=config,
        attn_implementation="flash_attention_2" if use_flash_attn else "eager",
        torch_dtype=dtype, trust_remote_code=trust_remote_code, **kwargs)


def build_student_config(teacher_name_or_path, selection, block_size=None, trust_remote_code=False):
    """Teacher config with fewer layers. For RoPE models ``max_position_embeddings`` is set to the
    training sequence length, as in the paper experiments."""
    config = AutoConfig.from_pretrained(teacher_name_or_path, trust_remote_code=trust_remote_code)
    arch = get_arch(config)
    setattr(config, arch.num_layers_attr, len(selection))
    setattr(config, "selected_layer_idx", list(selection))
    if arch.rotary and block_size is not None:
        setattr(config, "max_position_embeddings", block_size)
    return config


def student_from_config(config, arch: ArchSpec, use_flash_attn: bool, dtype=None, trust_remote_code=False):
    if arch.masked_lm:
        return AutoModelForMaskedLM.from_config(config=config, trust_remote_code=trust_remote_code)
    return AutoModelForCausalLM.from_config(
        config=config, attn_implementation="flash_attention_2" if use_flash_attn else "eager",
        torch_dtype=dtype, trust_remote_code=trust_remote_code)


def init_student_from_teacher(teacher, student, arch: ArchSpec, selection):
    """Copy teacher weights into the student: student layer ``i`` <- teacher layer ``selection[i]``,
    every non-layer parameter (embeddings, final norm, head) is copied as is."""
    teacher_state_dict = teacher.state_dict()
    student_state_dict = student.state_dict()
    prefix = arch.layer_prefix
    for key in student_state_dict.keys():
        if key.startswith(prefix):
            student_idx, rest = key[len(prefix):].split(".", 1)
            teacher_key = f"{prefix}{selection[int(student_idx)]}.{rest}"
        else:
            teacher_key = key
        student_state_dict[key] = teacher_state_dict[teacher_key].clone().detach()
    student.load_state_dict(student_state_dict)
    return student
