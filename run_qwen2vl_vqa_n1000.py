#!/usr/bin/env python3
"""Run Qwen2-VL WISE VQA baseline and enhancement on first 1000 records."""
import json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;N=1000
RUNS=[('baseline','hparams/WISE/qwen2vl_vqa_baseline.yaml'),('best_lar_target_w0p5','hparams/WISE/qwen2vl_vqa_best_lar_target.yaml')]
RROOT=ROOT/'results/QWEN_WISE_VQA_N1000';LROOT=ROOT/'run_logs/wise_vqa_qwen2vl_n1000'
def run(rid,cfg):
 out=RROOT/rid;ld=LROOT/rid;out.mkdir(parents=True,exist_ok=True);ld.mkdir(parents=True,exist_ok=True)
 env=os.environ.copy();env.update({'OMP_NUM_THREADS':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONPATH':str(ROOT),'PMRL_TASK':'VQA','PMRL_TEST_SIZE':str(N),'PMRL_SEED':'42','PMRL_WISE_HPARAMS':cfg,'PMRL_OUTPUT_DIR':str(out.relative_to(ROOT))});start=time.time()
 with (ld/'run.log').open('w') as f:p=subprocess.run(['/root/miniconda3/envs/easyedit/bin/python','run_wise_qwen2vl_ic.py'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
 (ld/'status.txt').write_text(f'WALL_SECONDS={time.time()-start}\nEXIT_CODE={p.returncode}\n')
 if p.returncode:raise RuntimeError(f'{rid} failed')
 d=json.loads((out/'result.json').read_text());assert d['task']=='VQA' and d['sample_count']==len(d['per_case'])==N;return d
def main():
 results={}
 for rid,cfg in RUNS:
  p=RROOT/rid/'result.json'
  if p.exists():
   d=json.loads(p.read_text())
   if d.get('sample_count')==N and len(d.get('per_case',[]))==N:results[rid]=d;continue
  results[rid]=run(rid,cfg)
 p=RROOT/'summary.json';p.write_text(json.dumps(results,indent=2));print(json.dumps({'status':'completed','baseline':results['baseline']['metrics'],'enhanced':results['best_lar_target_w0p5']['metrics'],'summary':str(p)}))
if __name__=='__main__':main()
