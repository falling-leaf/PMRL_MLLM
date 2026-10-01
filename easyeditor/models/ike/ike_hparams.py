from dataclasses import dataclass
from typing import List, Optional
import yaml

import torch

from ...util.hparams import HyperParams


@dataclass
class IKEHyperParams(HyperParams):

    # Module templates
    device: int
    alg_name: str
    model_name: str
    sentence_model_name: str = "./hugging_cache/all-MiniLM-L6-v2"

    # Method
    k: int = 16 # K icl examples
    results_dir: str = "./results"
    use_icl_examples: bool = True

    model_parallel: bool = False

    # Multimodal (Qwen2-VL / LLaVA-OneVision). Unused by the text IKE path.
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
    task_name: Optional[str] = None

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):

        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().resolve_config_paths(super().construct_float_from_scientific_notation(config))
        if isinstance(config.get("dtype"), str):
            config["dtype"] = getattr(torch, config["dtype"].replace("torch.", ""))

        assert (config and config['alg_name'] == 'IKE') or print(f'IKEHyperParams can not load from {hparams_name_or_path}, '
                                                f'alg_name is {config["alg_name"]} ')
        return cls(**config)


@dataclass
class PreEditHyperParams(HyperParams):
    """Config for scoring the original model. No update is applied."""

    alg_name: str
    model_name: str
    device: int
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
    model_parallel: bool = False
    results_dir: str = "./results"

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):
        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'
        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().resolve_config_paths(super().construct_float_from_scientific_notation(config))
        if isinstance(config.get("dtype"), str):
            config["dtype"] = getattr(torch, config["dtype"].replace("torch.", ""))
        assert (config and config['alg_name'] == 'preedit') or print(
            f'PreEditHyperParams can not load from {hparams_name_or_path}, '
            f'alg_name is {config["alg_name"]} ')
        return cls(**config)


@dataclass
class IKEMultimodalHyperParams(HyperParams):
    # Method
    k: int # K icl examples
    results_dir: str

    # Module templates
    device: int
    name: str
    alg_name: str
    model_name: str
    tokenizer_class: str
    tokenizer_name: str
    sentence_model_name: str

    ## Multimodal
    task_name: str
    qformer_checkpoint: str
    qformer_name_or_path: str
    state_dict_file: str
    
    # Image_dir
    coco_image: str
    rephrase_image: str
    exact_match: bool = False
    pretrained_ckpt: Optional[str] = None
    dtype: torch.dtype = torch.float32
    sequential_edit: bool = False
    use_chat_template: bool = True
    file_type: str = "image"
    qwen_max_pixels: int = 1280 * 28 * 28
    model_parallel: bool = False

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):

        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().resolve_config_paths(super().construct_float_from_scientific_notation(config))
        if isinstance(config.get("dtype"), str):
            config["dtype"] = getattr(torch, config["dtype"].replace("torch.", ""))

        assert (config and config['alg_name'] == 'IKE') or print(f'IKEMultimodalHyperParams can not load from {hparams_name_or_path}, '
                                                f'alg_name is {config["alg_name"]} ')
        return cls(**config)
