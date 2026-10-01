"""Evaluate the saved LLaVA-OV MEND IC ASAM checkpoint at step 15000."""
import json
from pathlib import Path
import torch
from easyeditor import CaptionDataset, MultimodalTrainer
from easyeditor.trainer.training_hparams.mend_multimodal_training_hparams import MENDMultimodalTrainingHparams

CONFIG = "hparams/TRAINING/MEND/llavaov-7b-lap-pmrl-ic-full.yaml"
ARCHIVE = "results/MEND_LLAVAOV_LAP_PMRL_IC_META_TRAIN_RETRY3/models/MEND/llava-onevision"
OUT = "results/MEND_LLAVAOV_IC_LAP_PMRL_STEP15000_N100"

def main():
    h = MENDMultimodalTrainingHparams.from_hparams(CONFIG)
    h.eval_only = True; h.archive = ARCHIVE; h.results_dir = OUT
    h.max_iters = 0; h.val_steps = 100; h.final_eval = True
    h.save = False; h.silent = False; h.verbose = True; h.debug = False
    torch.manual_seed(h.seed); torch.cuda.manual_seed_all(h.seed)
    val = CaptionDataset('/root/MMEdit/editing-data/caption/caption_eval_edit.json', config=h, size=100)
    train = CaptionDataset('/root/MMEdit/editing-data/caption/caption_train_edit.json', config=h, size=1)
    if len(val) != 100: raise RuntimeError(f'Expected 100 validation records, got {len(val)}')
    print(f'EVAL_START archive={ARCHIVE} size={len(val)} shift=True', flush=True)
    info = MultimodalTrainer(h, train, val).validate(log=True, steps=100)
    out = Path(OUT); out.mkdir(parents=True, exist_ok=True)
    payload = {'archive': str(Path(ARCHIVE).resolve()), 'step': 15000, 'sample_count': 100,
               'results': {k: float(v) if isinstance(v, (float, torch.Tensor)) else v for k,v in info.items()}}
    (out/'result.json').write_text(json.dumps(payload, indent=2))
    print('EVAL_DONE', flush=True)
    for k,v in info.items(): print(f'{k}: {v}', flush=True)
    print(f'RESULT={out.resolve()/"result.json"}', flush=True)
if __name__ == '__main__': main()
