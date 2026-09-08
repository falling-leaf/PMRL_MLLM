"""Hyperparameters for Transformer-Patcher multimodal editing."""

from dataclasses import dataclass
from typing import List, Optional

import torch
import yaml

from ...util.hparams import HyperParams


@dataclass
class TransformerPatcherMultimodalHyperParams(HyperParams):
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
    device: int
    alg_name: str
    model_name: str
    name: str
    tokenizer_class: str
    tokenizer_name: str
    dtype: torch.dtype
    # PMRL/LAP enhancement is explicitly gated to preserve baseline behavior.
    using_extra: bool = False
    using_lap: bool = False
    using_pmrl: bool = False
    using_image_embedding: bool = True
    num_rephrase: int = 5
    lap_epsilon: float = 1e-3
    pmrl_tau_alignment: float = 0.05
    pmrl_tau_regularization: float = 0.1
    pmrl_alignment_weight: float = 1.0
    pmrl_regularization_weight: float = 0.1
    pmrl_scale: float = 0.01
    lar_random_start: bool = False
    lar_pgd_steps: int = 1
    lar_step_size: float = 1e-3
    # HF multimodal models can perturb every non-target context token, including image features.
    lar_joint_perturbation: bool = False
    # Apply edit-target CE to each local view; never consumes image-rephrase labels.
    lar_target_loss_weight: float = 0.0
    # Shared HF tokenizer requires an explicit target-only-label protocol.
    objective_optimization: str = "only_label"
    adam_eps: float = 1e-8
    export_lap_samples: bool = False
    batch_size: int = 1
    max_length: int = 30
    model_parallel: bool = False
    use_chat_template: bool = False

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):
        if not hparams_name_or_path.endswith(".yaml"):
            hparams_name_or_path += ".yaml"
        with open(hparams_name_or_path, encoding="utf-8") as stream:
            config = HyperParams.construct_float_from_scientific_notation(yaml.safe_load(stream))
        dtype = config.get("dtype")
        if isinstance(dtype, str):
            dtype_name = dtype.removeprefix("torch.")
            if not hasattr(torch, dtype_name):
                raise ValueError(f"Unsupported dtype: {dtype}")
            config["dtype"] = getattr(torch, dtype_name)
        if config.get("alg_name") != "Transformer-Patcher":
            raise ValueError("TransformerPatcher hyperparameters require alg_name=Transformer-Patcher")
        if int(config.get("add_neuron_num", 0)) < 1:
            raise ValueError("add_neuron_num must be positive")
        count = len(config.get("inner_params", []))
        if count not in (2, 3):
            raise ValueError("inner_params must declare paired fc1/fc2 or SwiGLU gate/up/down weights")
        return cls(**config)
