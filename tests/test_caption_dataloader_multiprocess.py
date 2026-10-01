"""Regression tests for the forked-DataLoader image-handle bug.

The first 30 000-step LLaVA-OV MEND LAP+PMRL (ASAM) IC run died at step ~1004
with ``OSError: image file is truncated``.  Cause: ``CaptionDataset.__init__``
stored lazily-opened ``PIL.Image`` objects (file descriptors open).  ``DataLoader``
forks those descriptors into every worker, so all workers share one file offset
per image while each keeps its own empty decode cache: a record loaded by worker
A in epoch 1 made worker B's epoch-2 re-read start at EOF.

These tests pin both halves of the fix:
  1. every image stored in a dataset item is decoded and detached from its file,
  2. a multi-worker collate run crosses the epoch boundary it used to die on.
"""

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from PIL import Image  # noqa: E402

TRAIN_JSON = "/root/MMEdit/editing-data/caption/caption_train_edit.json"
CONFIG = str(REPO / "hparams/TRAINING/MEND/llavaov-7b-lap-pmrl-ic-graphfix-final2.yaml")

pytestmark = pytest.mark.skipif(
    not Path(TRAIN_JSON).exists() or not Path(CONFIG).exists(),
    reason="caption dataset / config not available on this box",
)


def _dataset(size):
    from easyeditor import CaptionDataset
    from easyeditor.trainer.training_hparams.mend_multimodal_training_hparams import (
        MENDMultimodalTrainingHparams,
    )

    hp = MENDMultimodalTrainingHparams.from_hparams(CONFIG)
    return CaptionDataset(TRAIN_JSON, config=hp, size=size)


def test_dataset_images_are_detached_from_files():
    """No item may keep an open file handle after construction."""
    ds = _dataset(20)
    seen = 0
    for item in ds:
        for key in ("image", "image_rephrase", "multimodal_locality_image"):
            img = item[key]
            if isinstance(img, Image.Image):
                # a detached image has no open file: ``copy()`` returns a plain
                # Image.Image, so ``fp`` is absent rather than None
                assert getattr(img, "fp", None) is None, f"{key} still holds an open file handle"
                # force a decode: a detached image needs no file access
                assert isinstance(img.tobytes(), bytes)
                seen += 1
    assert seen >= 60, f"expected 3 images per item, saw {seen}"


def test_multiworker_collate_crosses_epoch_boundary():
    """The epoch boundary is where the shared-offset crash used to trigger."""
    from torch.utils.data import DataLoader

    ds = _dataset(200)
    loader = DataLoader(
        ds, batch_size=1, shuffle=True, collate_fn=ds.collate_fn,
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2,
    )
    batches = 0
    for _ in range(2):  # two epochs => each record may change worker
        for _ in loader:
            batches += 1
    assert batches == 400
