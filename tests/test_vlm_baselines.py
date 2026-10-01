"""Baseline wiring for preedit, SERAC, IKE and FT on Qwen2-VL and LLaVA-OV.

These tests never load a 7B checkpoint and never import the ``easyeditor``
package.  That package currently has a pre-existing circular import through
``easyeditor.trainer``.  The modules under test are loaded by file path.
"""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]


def _package(name, path):
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    module.__package__ = name
    sys.modules[name] = module
    return module


def _load(qualified, relative):
    spec = importlib.util.spec_from_file_location(qualified, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = qualified.rpartition(".")[0]
    sys.modules[qualified] = module
    spec.loader.exec_module(module)
    return module


# Parent packages so the relative imports inside the baseline modules resolve
# without executing easyeditor/__init__.py or easyeditor/trainer/__init__.py.
for name, folder in (
    ("easyeditor", "easyeditor"),
    ("easyeditor.util", "easyeditor/util"),
    ("easyeditor.models", "easyeditor/models"),
    ("easyeditor.models.ft", "easyeditor/models/ft"),
    ("easyeditor.models.ike", "easyeditor/models/ike"),
    ("easyeditor.models.serac", "easyeditor/models/serac"),
    ("easyeditor.evaluate", "easyeditor/evaluate"),
):
    _package(name, ROOT / folder)

_load("easyeditor.util.hparams", "easyeditor/util/hparams.py")
_load("easyeditor.util.nethook", "easyeditor/util/nethook.py")
ft_hparams = _load("easyeditor.models.ft.ft_hparams", "easyeditor/models/ft/ft_hparams.py")
ike_hparams = _load("easyeditor.models.ike.ike_hparams", "easyeditor/models/ike/ike_hparams.py")
serac_hparams = _load(
    "easyeditor.models.serac.serac_multimodal_hparams",
    "easyeditor/models/serac/serac_multimodal_hparams.py",
)
ike_main = _load("easyeditor.models.ike.ike_main", "easyeditor/models/ike/ike_main.py")
ft_main = _load("easyeditor.models.ft.ft_main", "easyeditor/models/ft/ft_main.py")

FTHyperParams = ft_hparams.FTHyperParams
IKEHyperParams = ike_hparams.IKEHyperParams
PreEditHyperParams = ike_hparams.PreEditHyperParams
SERACMultimodalHparams = serac_hparams.SERACMultimodalHparams
_hf_supervised_batch = ft_main._hf_supervised_batch
_is_hf_multimodal = ft_main._is_hf_multimodal
apply_ike_to_multimodal_model = ike_main.apply_ike_to_multimodal_model


def default_config(method, model):
    folder = {"preedit": "preedit", "serac": "SERAC", "ike": "IKE", "ft": "FT"}[method]
    name = model if method == "serac" else ("llavaov_7b" if model == "llavaov" else f"{model}_7b")
    return ROOT / "hparams" / folder / f"{name}.yaml"


MODELS = ("qwen2vl", "llavaov")


def test_four_baselines_resolve_configs_for_both_models():
    for model in MODELS:
        for method in ("preedit", "serac", "ike", "ft"):
            path = default_config(method, model)
            assert path.exists(), path
            loaded = yaml.safe_load(path.read_text())
            assert "using_asam" not in loaded
            assert loaded.get("using_extra") is not True


def test_hparams_load_without_touching_model_weights():
    for model in MODELS:
        stem = "llavaov_7b" if model == "llavaov" else f"{model}_7b"
        pre = PreEditHyperParams.from_hparams(str(ROOT / f"hparams/preedit/{stem}.yaml"))
        assert pre.alg_name == "preedit"

        ike = IKEHyperParams.from_hparams(str(ROOT / f"hparams/IKE/{stem}.yaml"))
        assert ike.alg_name == "IKE"
        assert ike.k == 0

        ft = FTHyperParams.from_hparams(str(ROOT / f"hparams/FT/{stem}.yaml"))
        assert ft.alg_name == "FT"
        assert _is_hf_multimodal(ft)
        assert ft.rewrite_module_tmp.format(ft.layers[0]).startswith("model.language_model")

        serac_name = "qwen2vl" if model == "qwen2vl" else "llavaov"
        serac = SERACMultimodalHparams.from_hparams(str(ROOT / f"hparams/SERAC/{serac_name}.yaml"))
        assert serac.alg_name == "SERAC_MULTI"
        train_yaml = yaml.safe_load((ROOT / f"hparams/TRAINING/SERAC/{serac_name}.yaml").read_text())
        assert train_yaml["alg"] == "SERAC_MULTI"
        assert train_yaml["archive"] is None


def test_registries_keep_preedit_out_and_ft_in():
    text = (ROOT / "easyeditor/util/alg_dict.py").read_text()
    assert "'FT': apply_ft_to_model" in text
    assert "'preedit'" not in text
    editor = (ROOT / "easyeditor/editors/multimodal_editor.py").read_text()
    assert "self.alg_name == 'preedit'" in editor
    assert "request['target_new'] = request['target']" in editor


def test_ike_k0_returns_only_the_edited_fact():
    hparams = SimpleNamespace(k=0, device=0)
    request = {"prompt": "What color?", "target": "blue"}
    examples = apply_ike_to_multimodal_model(None, None, request, hparams, train_ds=None)
    assert examples == ["New Fact: What color? blue\nPrompt: What color? blue\n\n"]


def test_ft_multimodal_batch_uses_the_evaluator_batch(monkeypatch):
    hparams = SimpleNamespace(model_name="qwen2-vl", device=torch.device("cpu"), dtype=torch.float32)
    sentinel = {
        "multimodal_inputs": {"input_ids": torch.tensor([[1, 2, 3]])},
        "labels": torch.tensor([[-100, -100, 9]]),
    }

    def _fake_prepare(*args, **kwargs):
        return sentinel

    import importlib
    evaluator = types.ModuleType("easyeditor.evaluate.multimodal_evaluate")
    evaluator.prepare_multimodal_hf_edit = _fake_prepare
    monkeypatch.setitem(sys.modules, "easyeditor.evaluate.multimodal_evaluate", evaluator)
    batch = _hf_supervised_batch(object(), "prompt", "blue", object(), hparams)
    assert torch.equal(batch["labels"], sentinel["labels"])


def test_text_ft_yaml_still_loads():
    hp = FTHyperParams.from_hparams(str(ROOT / "hparams/FT/llama-7b.yaml"))
    assert hp.alg_name == "FT"
    assert not _is_hf_multimodal(hp)
