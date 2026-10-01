from dataclasses import dataclass
from ...util.hparams import HyperParams
from typing import Optional, Any, List
import yaml
import torch


@dataclass
class MENDMultimodalTrainingHparams(HyperParams):
    
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
    batch_size: int
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

    qformer_checkpoint: str
    exact_match: bool = False
    model_parallel: bool = False
    freeze_qformer: bool = True
    max_epochs: Optional[int] = None
    max_iters: Optional[int] = None  
    pretrained_ckpt: Optional[str] = None  
    dtype: torch.dtype = torch.bfloat16

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
    pmrl_visual_pooling: bool = False
    lar_target_loss_weight: float = 0.0
    mend_extra_at_eval: bool = False
    qwen_max_pixels: int = 1280 * 28 * 28
    mend_log_grad_diagnostics: bool = False
    checkpoint_before_validation: bool = False

    # Outer-loop ASAM on the MEND hypernetwork.  Defaults keep the baseline
    # inner-edit / outer-loss recipe unchanged.
    using_asam: bool = False
    asam_epsilon: float = 0.002
    asam_scale_rho: float = 0.1
    asam_replace: bool = True
    # UniKE's rho floor is for zero-init adapters.  MEND edit_lrs start at
    # ~1e-4, so the same floor inflates their SAM neighborhood ~1000x and
    # lets them dominate ||T g||.  Default: perturb the hypernetwork only.
    asam_exclude_edit_lrs: bool = True
    baseline_t_loc: Optional[float] = None
    baseline_m_loc: Optional[float] = None

    # Throughput knobs.  Defaults reproduce the original behaviour exactly;
    # see docs/MEND_LLAVAOV_IC_PERF.md for measured effects.
    dataloader_num_workers: int = 0
    gpu_cache_free_threshold_gb: float = 8.0
    mend_log_weight_diagnostics: bool = False
    kl_chunk_size: int = 32

    # Train-loss overfit stop.  Off by default so existing MEND runs are
    # unchanged.  When on, max_iters is also capped at overfit_stop_hard_max.
    overfit_stop: bool = False
    overfit_stop_key: str = "loss/total_train"
    overfit_stop_min_step: int = 4000
    overfit_stop_hard_max: int = 10000
    overfit_stop_patience_logs: int = 3
    overfit_stop_rel_rebound: float = 0.25
    overfit_stop_smooth_window: int = 3
    
    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):

        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().resolve_config_paths(super().construct_float_from_scientific_notation(config))
        if isinstance(config.get("dtype"), str):
            config["dtype"] = getattr(torch, config["dtype"].replace("torch.", ""))

        assert (config and config['alg'] == 'MEND') or print(f'MENDMultimodalTrainingHyperParams can not load from {hparams_name_or_path}, '
                                                f'alg_name is {config["alg"]} ')
        return cls(**config)

