from dataclasses import dataclass
from typing import List, Union, Optional, Any
from ...util.hparams import HyperParams
import yaml
import torch

@dataclass
class WISEMultimodalHyperParams(HyperParams):
    # Multimodal
    qformer_name_or_path: str
    qformer_checkpoint: str
    state_dict_file: str
    hidden_act: str
    pretrained_ckpt: str
    
    # Image_dir
    file_type: str # single image or video
    exact_match: bool
    coco_image: str
    rephrase_image: str

    # Experiments
    sequential_edit: bool
    edit_lr: float
    n_iter: int
    # Method
    objective_optimization: str
    mask_ratio: float
    alpha: float    # act_margin[0]
    beta: float  # act_margin[1]
    gamma: float  # act_margin[2]
    act_ratio: float
    merge_freq: int
    retrieve: bool
    replay: bool
    save_freq: Union[int, None]
    merge_alg: str
    norm_constraint: float
    # Module templates
    inner_params: List[str]
    weights: Union[float, None]
    densities: Union[float, None]

    device: int
    alg_name: str
    model_name: str
    name: str
    tokenizer_class: str
    tokenizer_name: str
    dtype: torch.dtype

    # Optional LAP + PMRL enhancement. When using_extra is false the WISE
    # optimization and evaluation paths remain baseline-only.
    using_extra: bool = False
    using_lap: bool = False
    using_pmrl: bool = False
    using_image_embedding: bool = False
    num_rephrase: int = 5
    lap_epsilon: float = 1e-3
    pmrl_tau_alignment: float = 0.05
    pmrl_tau_regularization: float = 0.1
    pmrl_alignment_weight: float = 1.0
    pmrl_regularization_weight: float = 0.1
    pmrl_scale: float = 1.0
    lar_random_start: bool = False
    lar_pgd_steps: int = 1
    lar_step_size: float = 1e-3
    lar_joint_perturbation: bool = False
    lar_target_loss_weight: float = 0.0
    pmrl_visual_pooling: bool = False
    pmrl_spectral_alignment: bool = False
    export_lap_samples: bool = False

    # Defaults
    batch_size: int = 1
    max_length: int = 30
    model_parallel: bool = False
    use_chat_template: bool = False

    # Save and Load
    save_path: str = None
    load_path: str = None

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):
        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert config['merge_freq'] % config['save_freq'] == 0, 'merge_freq need to be divisible by save_freq (like 1000 / 500)'
        assert len(config['act_margin']) == 3
        config['alpha'], config['beta'], config['gamma'] = config['act_margin'][0], config['act_margin'][1], config['act_margin'][2]
        config.pop('act_margin')

        if 'dtype' in config:
            dtype_val = config['dtype']
            if isinstance(dtype_val, str):
                # 移除可能存在的 "torch." 前缀 (例如 YAML 里写了 "torch.bfloat16")
                dtype_name = dtype_val.replace('torch.', '')
                # 从 torch 模块获取对应的属性
                if hasattr(torch, dtype_name):
                    config['dtype'] = getattr(torch, dtype_name)
                else:
                    raise ValueError(f"Unsupported dtype: {dtype_val}")

        assert (config and config['alg_name'] == 'WISE'), \
            f'WISEHyperParams can not load from {hparams_name_or_path}. alg_name is {config["alg_name"]}'
        return cls(**config)