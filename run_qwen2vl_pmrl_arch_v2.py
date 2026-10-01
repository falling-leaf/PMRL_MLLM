#!/usr/bin/env python3
"""Screen PMRL architecture-v2 loss weights, then confirm best on N=100."""
import json, os, subprocess, time
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parent
BASE=yaml.safe_load((ROOT/'hparams/WISE/qwen2vl_ic_lap_pmrl.yaml').read_text())
RUNS=[
 ('arch-v2-a1-r0',20,{'pmrl_alignment_weight':1.0,'pmrl_regularization_weight':0.0,'pmrl_scale':0.01}),
 ('arch-v2-a1-r0p1',20,{'pmrl_alignment_weight':1.0,'pmrl_regularization_weight':0.1,'pmrl_scale':0.01}),
 ('arch-v2-a1-r0p5',20,{'pmrl_alignment_weight':1.0,'pmrl_regularization_weight':0.5,'pmrl_scale':0.01}),
]
CONFIG_DIR=ROOT/'hparams/WISE/qwen2vl_arch_v2'
RESULT_ROOT=ROOT/'results/QWEN_WISE_IC_PMRL_ARCH_V2_RUNS'
LOG_ROOT=ROOT/'run_logs/wise_ic_qwen2vl_arch_v2'

def run(run_id,n,overrides):
 cfg=dict(BASE);cfg.update(overrides)
 CONFIG_DIR.mkdir(parents=True,exist_ok=True);RESULT_ROOT.mkdir(parents=True,exist_ok=True);LOG_ROOT.mkdir(parents=True,exist_ok=True)
 cp=CONFIG_DIR/f'{run_id}.yaml';cp.write_text(yaml.safe_dump(cfg,sort_keys=False))
 out=RESULT_ROOT/run_id;log=LOG_ROOT/run_id;out.mkdir(parents=True,exist_ok=True);log.mkdir(parents=True,exist_ok=True)
 env=os.environ.copy();env.update({'OMP_NUM_THREADS':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONPATH':str(ROOT),'PMRL_TEST_SIZE':str(n),'PMRL_SEED':'42','PMRL_WISE_HPARAMS':str(cp.relative_to(ROOT)),'PMRL_OUTPUT_DIR':str(out.relative_to(ROOT))})
 start=time.time()
 with (log/'run.log').open('w') as f: p=subprocess.run(['/root/miniconda3/envs/easyedit/bin/python','run_wise_qwen2vl_ic.py'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
 (log/'status.txt').write_text(f'WALL_SECONDS={time.time()-start}\nEXIT_CODE={p.returncode}\n')
 if p.returncode: raise SystemExit(f'{run_id} failed')
 d=json.loads((out/'result.json').read_text());return {'run_id':run_id,'n':n,**overrides,**d['metrics'],'wall_seconds':d['wall_seconds'],'result_path':str((out/'result.json').relative_to(ROOT))}

def main():
 rows=[run(*spec) for spec in RUNS]
 best=max(rows,key=lambda r:(r['gen_avg'],r['rephrase_image_acc']))
 confirm=run('arch-v2-confirm-'+best['run_id'],100,{k:best[k] for k in ('pmrl_alignment_weight','pmrl_regularization_weight','pmrl_scale')})
 rows.append(confirm)
 path=ROOT/'results/QWEN_WISE_IC_PMRL_ARCH_V2_SUMMARY.json';path.write_text(json.dumps(rows,indent=2))
 print(json.dumps({'best_screen':best['run_id'],'confirm':confirm,'summary':str(path)}))
if __name__=='__main__':main()
