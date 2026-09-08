from dataclasses import dataclass
from ...util.hparams import HyperParams
from .mend_hparams import MENDHyperParams
from typing import Optional, Any, List
import yaml
import torch


@dataclass
class MENDMultimodalHparams(HyperParams):
    
    # Multimodal
    qformer_name_or_path: str
    state_dict_file: str
    
    # Image_dir
    coco_image: str
    rephrase_image: str
    
    # Model
    name: str
    model_name: str
    model_class: str
    tokenizer_class: str
    tokenizer_name: str
    inner_params: List[str]

    archive: Any

    # Method
    alg: str
    lr: float
    edit_lr: float
    lr_lr: float
    lr_scale: float
    seed: int
    debug: bool
    cedit: float
    iedit: float
    cloc: float
    cbase: float
    dropout: float
    train_base: bool
    no_grad_layers: Any
    one_sided: bool
    n_hidden: int
    hidden_dim: Any
    init: str
    norm: bool
    combine: bool
    x_only: bool
    delta_only: bool
    act: str
    rank: int
    mlp_class: str
    shared: bool

    # Output

    results_dir: str

    # Train
    device: str
    model_save_pt: int
    silent: bool
    log_interval: int
    eval_log_interval:int
    final_eval:bool
    val_interval: int
    early_stop_patience: int
    early_stop_key: str
    eval_only: bool
    half: bool
    save: bool
    verbose: bool

    val_batch_size: int
    accumulate_bs: int
    val_steps: int
    opt: str
    grad_clip: float

    alg_name: str
    
    exact_match: bool = False
    batch_size: int = 1
    max_length: int = 30
    max_epochs: Optional[int] = None
    max_iters: Optional[int] = None
    model_parallel: bool = False
    qformer_checkpoint: Optional[str] = None
    freeze_qformer: bool = True
    pretrained_ckpt: Optional[str] = None  
    dtype: torch.dtype = torch.bfloat16
    sequential_edit: bool = False

    # Optional MEND enhancement. The baseline is strictly isolated unless all
    # three gates are enabled.
    using_extra: bool = False
    using_lap: bool = False
    using_pmrl: bool = False
    using_image_embedding: bool = True
    num_rephrase: int = 5
    lap_epsilon: float = 0.001
    pmrl_tau_alignment: float = 0.05
    pmrl_tau_regularization: float = 0.1
    pmrl_alignment_weight: float = 1.0
    pmrl_regularization_weight: float = 0.1
    pmrl_scale: float = 1.0
    mend_log_grad_diagnostics: bool = False
    
    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):

        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)
        if isinstance(config.get("dtype"), str):
            config["dtype"] = getattr(torch, config["dtype"].replace("torch.", ""))

        assert (config and config['alg'] == 'MEND') or print(f'MENDMultimodalHyperParams can not load from {hparams_name_or_path}, '
                                                f'alg_name is {config["alg"]} ')
        return cls(**config)

