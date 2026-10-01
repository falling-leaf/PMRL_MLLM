import importlib.util
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]


def _load_utils():
    spec = importlib.util.spec_from_file_location(
        "ee_utils_ckpt", ROOT / "easyeditor/trainer/utils.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


utils = _load_utils()


def _checkpoint_payload(host, stats=None):
    stopper = getattr(host, "stopper", None)
    return {
        "model": host.model.state_dict(),
        "opt": host.opt.state_dict(),
        "lr_opt": None,
        "val_stats": stats,
        "start_time": host.start_time,
        "elapsed_time": 0.0,
        "step": host.global_iter,
        "stopper": stopper.state_dict() if stopper is not None else None,
        "rng_state": None,
    }


def _save_periodic(host):
    payload = _checkpoint_payload(host, None)
    last_path = f"{host.save_path}.last"
    step_path = f"{host.save_path}.step{host.global_iter}"
    utils.atomic_torch_save(payload, last_path)
    utils.atomic_torch_save(payload, step_path)
    numbered = []
    for path in Path(host.save_path).parent.glob(Path(host.save_path).name + ".step*"):
        suffix = path.name.split(".step")[-1]
        if suffix.isdigit():
            numbered.append((int(suffix), path))
    numbered.sort()
    for _, path in numbered[:-3]:
        path.unlink()


class _TinyCkptHost:
    def __init__(self, save_path):
        self.model = nn.Linear(3, 3, bias=False)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.global_iter = 12
        self.start_time = utils.formatted_timestamp()
        self.stopper = utils.EarlyStopper(7, "acc/loc_floored_val")
        self.stopper.best_value = 2.5
        self.stopper.best_iter = 10
        self.stopper.current_iter = 12
        self.save_path = str(save_path)


def test_periodic_checkpoint_roundtrip(tmp_path):
    host = _TinyCkptHost(tmp_path / "llava-onevision")
    with torch.no_grad():
        host.model.weight.fill_(0.42)
    _save_periodic(host)
    last = Path(str(host.save_path) + ".last")
    step = Path(str(host.save_path) + ".step12")
    assert last.exists()
    assert step.exists()
    archive, _ = utils.load_archive(str(last))
    assert archive["step"] == 12
    assert archive["stopper"]["best_iter"] == 10
    restored = nn.Linear(3, 3, bias=False)
    restored.load_state_dict(archive["model"])
    assert torch.allclose(restored.weight, host.model.weight)
    restored_stopper = utils.EarlyStopper(7, "acc/loc_floored_val")
    restored_stopper.load_state_dict(archive["stopper"])
    assert restored_stopper.best_value == 2.5
    assert restored_stopper.best_iter == 10


def test_save_periodic_keeps_last_three_step_files(tmp_path):
    host = _TinyCkptHost(tmp_path / "llava-onevision")
    for step in (2, 4, 6, 8):
        host.global_iter = step
        _save_periodic(host)
    numbered = sorted(p.name for p in tmp_path.glob("llava-onevision.step*"))
    assert numbered == [
        "llava-onevision.step4",
        "llava-onevision.step6",
        "llava-onevision.step8",
    ]


def test_initial_global_iter_reads_checkpoint_step():
    assert utils.initial_global_iter(None) == 0
    assert utils.initial_global_iter({"step": 15000}) == 15000


def test_run_does_not_reset_global_iter_to_zero():
    source = (ROOT / "easyeditor/trainer/BaseTrainer.py").read_text(encoding="utf-8")
    run_src = source.split("def run(self):", 1)[1]
    assert "self.global_iter = 0" not in run_src.split("def ")[0]
    assert "Resuming from step" in run_src


def test_atomic_save_is_loadable(tmp_path):
    path = tmp_path / "ckpt"
    utils.atomic_torch_save({"step": 3, "model": {"w": torch.tensor(1.0)}}, str(path))
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    assert loaded["step"] == 3


def test_early_stopper_acc_key_is_higher_better():
    stopper = utils.EarlyStopper(5, "acc/loc_floored_val")
    stats = {
        "acc/loc_floored_val": utils.loc_floored_score(
            {
                "edit/acc_val": 0.7,
                "image_rephrase/acc_val": 0.7,
                "loc/acc_val": 0.97,
                "image_loc/acc_val": 0.71,
            },
            t_floor=0.97,
            m_floor=0.71,
        )
    }
    assert stopper.update(5, stats) is True
    worse = dict(stats)
    worse["acc/loc_floored_val"] = -1.0
    assert stopper.update(10, worse) is False
