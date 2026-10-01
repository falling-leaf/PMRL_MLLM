#!/usr/bin/env python3
"""VQA N=100 budgeted tuning: lower target supervision weights."""
import json,os,subprocess,time
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parent;N=100
BASE=yaml.safe_load((ROOT/'hparams/WISE/qwen2vl_vqa_best_lar_target.yaml').read_text())
RUNS=[('target_w0p10',0.10),('target_w0p25',0.25)]
C=ROOT/'hparams/WISE/qwen2vl_vqa_tuning';R=ROOT/'results/QWEN_WISE_VQA_TUNING_N100';L=ROOT/'run_logs/wise_vqa_qwen2vl_tuning_n100'
def run(rid,w):
 cfg=dict(BASE);cfg['lar_target_loss_weight']=w;C.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True);L.mkdir(parents=True,exist_ok=True)
 cp=C/f'{rid}.yaml';cp.write_text(yaml.safe_dump(cfg,sort_keys=False));out=R/rid;ld=L/rid;out.mkdir(parents=True,exist_ok=True);ld.mkdir(parents=True,exist_ok=True)
 env=os.environ.copy();env.update({'OMP_NUM_THREADS':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONPATH':str(ROOT),'PMRL_TASK':'VQA','PMRL_TEST_SIZE':str(N),'PMRL_SEED':'42','PMRL_WISE_HPARAMS':str(cp.relative_to(ROOT)),'PMRL_OUTPUT_DIR':str(out.relative_to(ROOT))});start=time.time()
 with (ld/'run.log').open('w') as f:p=subprocess.run(['/root/miniconda3/envs/easyedit/bin/python','run_wise_qwen2vl_ic.py'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
 (ld/'status.txt').write_text(f'WALL_SECONDS={time.time()-start}\nEXIT_CODE={p.returncode}\n');assert p.returncode==0,rid;d=json.loads((out/'result.json').read_text());assert d['sample_count']==len(d['per_case'])==N;return {'run_id':rid,'target_weight':w,**d['metrics'],'wall_seconds':d['wall_seconds'],'result_path':str((out/'result.json').relative_to(ROOT))}
def main():
 start=time.time();rows=[]
 for rid,w in RUNS:
  if time.time()-start>5100:break
  rows.append(run(rid,w))
 p=R/'summary.json';p.write_text(json.dumps(rows,indent=2));print(json.dumps({'status':'completed','runs':rows,'summary':str(p)}))
if __name__=='__main__':main()
