from dataclasses import dataclass
from typing import List, Optional
import yaml

import torch

from ...util.hparams import HyperParams


@dataclass
class FTHyperParams(HyperParams):
    # Method
    layers: List[int]
    num_steps: int
    lr: float
    weight_decay: float
    kl_factor: float
    norm_constraint: float

    # Module templates
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str
    device: int
    alg_name: str
    model_name: str
    objective_optimization: str

    # Defaults
    batch_size: int = 64
    max_length: int = 40
    model_parallel: bool = False

    # Multimodal (Qwen2-VL / LLaVA-OneVision). Unused by the text FT path.
    name: Optional[str] = None
    tokenizer_name: Optional[str] = None
    tokenizer_class: Optional[str] = None
    coco_image: Optional[str] = None
    rephrase_image: Optional[str] = None
    qformer_name_or_path: Optional[str] = None
    qformer_checkpoint: Optional[str] = None
    state_dict_file: Optional[str] = None
    pretrained_ckpt: Optional[str] = None
    dtype: torch.dtype = torch.bfloat16
    sequential_edit: bool = False
    exact_match: bool = False
    use_chat_template: bool = True
    file_type: str = "image"
    qwen_max_pixels: int = 1280 * 28 * 28

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):

        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().resolve_config_paths(super().construct_float_from_scientific_notation(config))
        if isinstance(config.get("dtype"), str):
            config["dtype"] = getattr(torch, config["dtype"].replace("torch.", ""))

        assert (config and config['alg_name'] == 'FT') or print(f'FTHyperParams can not load from {hparams_name_or_path}, '
                                                f'alg_name is {config["alg_name"]} ')
        return cls(**config)
