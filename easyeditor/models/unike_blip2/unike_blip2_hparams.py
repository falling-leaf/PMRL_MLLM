"""Explicit, BLIP-2-only hyperparameters for the UniKE integration.

The original UniKE implementation only releases MiniGPT-4 checkpoints and
per-case latent memories.  This adapter keeps the published architectural
mechanism (paired FFN expansion + latent memory feature shift), while requiring
locally supplied BLIP-2-compatible memories rather than silently downloading or
reusing incompatible MiniGPT-4 artifacts.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
import yaml

from ...util.hparams import HyperParams


@dataclass
class UniKEBLIP2HyperParams(HyperParams):
    qformer_name_or_path: str
    qformer_checkpoint: str
    state_dict_file: str
    pretrained_ckpt: Optional[str]
    file_type: str
    exact_match: bool
    coco_image: str
    rephrase_image: str
    sequential_edit: bool
    edit_lr: float
    n_iter: int
    locality_weight: float
    add_neuron_num: int
    inner_params: List[str]
    l_ike_layers: List[str]
    device: int
    alg_name: str
    model_name: str
    name: str
    tokenizer_class: str
    tokenizer_name: str
    dtype: torch.dtype
    retrieved_knowledge_path: Optional[str] = None
    latent_ike_path: Optional[str] = None
    semantic_encoder_path: Optional[str] = None
    retrieval_top_k: int = 40
    feature_shift_scale: float = 1.0
    semantic_gate: bool = True
    adam_eps: float = 1e-4
    using_asam: bool = False
    asam_epsilon: float = 1e-3
    asam_weight: float = 1.0
    asam_visual_only: bool = True
    batch_size: int = 1
    max_length: int = 30
    model_parallel: bool = False
    use_chat_template: bool = False

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):
        path = Path(hparams_name_or_path)
        if path.suffix != ".yaml":
            path = path.with_suffix(".yaml")
        with path.open(encoding="utf-8") as stream:
            config = HyperParams.construct_float_from_scientific_notation(
                yaml.safe_load(stream)
            )
        dtype = config.get("dtype")
        if isinstance(dtype, str):
            dtype_name = dtype.removeprefix("torch.")
            if not hasattr(torch, dtype_name):
                raise ValueError(f"Unsupported dtype: {dtype}")
            config["dtype"] = getattr(torch, dtype_name)
        if config.get("alg_name") != "UniKE-BLIP2":
            raise ValueError("UniKE BLIP-2 hyperparameters require alg_name=UniKE-BLIP2")
        if config.get("model_name") != "blip2":
            raise ValueError("This UniKE integration is intentionally limited to BLIP-2")
        if len(config.get("inner_params", [])) != 2:
            raise ValueError("UniKE-BLIP2 needs exactly [fc1.weight, fc2.weight]")
        if int(config.get("add_neuron_num", 0)) < 1:
            raise ValueError("add_neuron_num must be positive")
        if not config.get("l_ike_layers"):
            raise ValueError("l_ike_layers must list the FFNs receiving latent feature shifts")
        return cls(**config)
