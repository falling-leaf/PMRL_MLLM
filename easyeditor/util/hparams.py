import json
import os
import re
from dataclasses import dataclass
from dataclasses import asdict
from pathlib import Path


# Machine-local roots that appear in checked-in YAML. They are rewritten to the
# directories selected by PMRL_MODEL_DIR / PMRL_DATA_DIR (or the conventional
# siblings of this repository) so a fresh clone does not embed /root paths.
_PATH_KEYS = {
    "name",
    "model_name",
    "tokenizer_name",
    "sentence_model_name",
    "small_name",
    "cls_name",
    "qformer_checkpoint",
    "qformer_name_or_path",
    "state_dict_file",
    "pretrained_ckpt",
    "coco_image",
    "rephrase_image",
    "archive",
    "results_dir",
    "retrieved_knowledge_path",
    "latent_ike_path",
    "semantic_encoder_path",
    "save_path",
    "load_path",
}
_ABS_ROOT = re.compile(r"^/(?:root|dev/shm)(?:/|$)")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def model_dir() -> Path:
    return Path(os.environ.get("PMRL_MODEL_DIR", repo_root().parent / "hugging_cache")).expanduser()


def data_dir() -> Path:
    return Path(os.environ.get("PMRL_DATA_DIR", repo_root().parent / "MMEdit")).expanduser()


def resolve_config_path(value):
    """Rewrite a YAML path onto this machine without touching method knobs.

    Absolute paths that already exist are kept. Known cache/data prefixes and
    repo-relative ``hugging_cache/...`` stems are remapped. Placeholders such as
    ``.`` and ``null`` are left alone.
    """
    if not isinstance(value, str):
        return value
    raw = value.strip()
    if raw in {"", ".", "..", "null", "None", "none"}:
        return value
    expanded = os.path.expanduser(raw)
    if os.path.isabs(expanded) and os.path.exists(expanded):
        return expanded
    normalized = expanded.replace("\\", "/")
    # /dev/shm/vicuna was a ramdisk alias of the Vicuna-7B checkpoint.
    if normalized in {"/dev/shm/vicuna", "/root/hugging_cache/vicuna-7b", "/root/hugging_cache/Vicuna"}:
        return str(model_dir() / "vicuna-7b-v1.5")
    prefixes = (
        ("/root/hugging_cache/", model_dir()),
        ("hugging_cache/", model_dir()),
        ("/root/MMEdit/", data_dir()),
        ("/root/PMRL_MLLM/", repo_root()),
    )
    for prefix, root in prefixes:
        if normalized.startswith(prefix):
            return str(root / normalized[len(prefix):])
    if normalized.startswith("/root/hugging_cache"):
        return str(model_dir())
    if normalized.startswith("/root/MMEdit"):
        return str(data_dir())
    if not os.path.isabs(normalized) and (repo_root() / normalized).exists():
        return str(repo_root() / normalized)
    if _ABS_ROOT.match(normalized):
        # Leave unknown absolute paths untouched so a missing download fails
        # with the original location instead of a silently rewritten one.
        return normalized
    return value


@dataclass
class HyperParams:
    """
    Simple wrapper to store hyperparameters for Python-based rewriting methods.
    """

    @classmethod
    def from_json(cls, fpath):
        with open(fpath, "r") as f:
            data = json.load(f)

        return cls(**data)

    def construct_float_from_scientific_notation(config: dict):
        for key, value in config.items():
            if isinstance(value, str):
                try:
                    # Convert scalar to float if it is in scientific notation format
                    config[key] = float(value)
                except:
                    pass
        return config
    
    def to_dict(config) -> dict:
        dict = asdict(config)
        return dict

    @staticmethod
    def resolve_config_paths(config: dict) -> dict:
        if not isinstance(config, dict):
            return config
        for key, value in list(config.items()):
            if key in _PATH_KEYS:
                config[key] = resolve_config_path(value)
        return config
            
        

    # @classmethod
    # def from_hparams(cls, hparams_name_or_path: str):
    #
    #     if '.yaml' not in hparams_name_or_path:
    #         hparams_name_or_path = hparams_name_or_path + '.yaml'
    #     config = compose(hparams_name_or_path)
    #
    #     assert config.alg_name in ALG_DICT.keys() or print(f'Editing Alg name {config.alg_name} not supported yet.')
    #
    #     params_class, apply_algo = ALG_DICT[config.alg_name]
    #
    #     return params_class(**config)
