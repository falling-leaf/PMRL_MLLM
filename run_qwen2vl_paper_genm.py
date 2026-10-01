#!/usr/bin/env python3
"""Paper-driven LAR/RCSL screening targeting Qwen IC Gen-M."""
import json, os, subprocess, time
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parent
BASE=yaml.safe_load((ROOT/'hparams/WISE/qwen2vl_ic_lap_pmrl.yaml').read_text())
RUNS=[
 ('paper-lar-rand',20,{'lar_random_start':True,'lar_pgd_steps':1,'lar_step_size':0.001,'lar_joint_perturbation':False,'pmrl_spectral_alignment':False,'pmrl_regularization_weight':0.5}),
 ('paper-lar-joint-rand',20,{'lar_random_start':True,'lar_pgd_steps':1,'lar_step_size':0.001,'lar_joint_perturbation':True,'pmrl_spectral_alignment':False,'pmrl_regularization_weight':0.5}),
 ('paper-lar-joint-rand-eps3',20,{'lar_random_start':True,'lar_pgd_steps':1,'lar_step_size':0.003,'lap_epsilon':0.003,'lar_joint_perturbation':True,'pmrl_spectral_alignment':False,'pmrl_regularization_weight':0.5}),
 ('paper-lar-rand-rcsl',20,{'lar_random_start':True,'lar_pgd_steps':1,'lar_step_size':0.001,'lar_joint_perturbation':False,'pmrl_spectral_alignment':True,'pmrl_regularization_weight':0.0}),
 ('paper-lar-joint-rcsl',20,{'lar_random_start':True,'lar_pgd_steps':1,'lar_step_size':0.001,'lar_joint_perturbation':True,'pmrl_spectral_alignment':True,'pmrl_regularization_weight':0.0}),
]
CROOT=ROOT/'hparams/WISE/qwen2vl_paper';RROOT=ROOT/'results/QWEN_WISE_IC_PAPER_GENM_RUNS';LROOT=ROOT/'run_logs/wise_ic_qwen2vl_paper_genm'
def run(rid,n,ov):
 cfg=dict(BASE);cfg.update(ov);CROOT.mkdir(parents=True,exist_ok=True);RROOT.mkdir(parents=True,exist_ok=True);LROOT.mkdir(parents=True,exist_ok=True)
 cp=CROOT/f'{rid}.yaml';cp.write_text(yaml.safe_dump(cfg,sort_keys=False));out=RROOT/rid;log=LROOT/rid;out.mkdir(parents=True,exist_ok=True);log.mkdir(parents=True,exist_ok=True)
 env=os.environ.copy();env.update({'OMP_NUM_THREADS':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONPATH':str(ROOT),'PMRL_TEST_SIZE':str(n),'PMRL_SEED':'42','PMRL_WISE_HPARAMS':str(cp.relative_to(ROOT)),'PMRL_OUTPUT_DIR':str(out.relative_to(ROOT))})
 start=time.time()
 with (log/'run.log').open('w') as f:p=subprocess.run(['/root/miniconda3/envs/easyedit/bin/python','run_wise_qwen2vl_ic.py'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
 (log/'status.txt').write_text(f'WALL_SECONDS={time.time()-start}\nEXIT_CODE={p.returncode}\n')
 if p.returncode:raise SystemExit(f'{rid} failed')
 d=json.loads((out/'result.json').read_text());return {'run_id':rid,'n':n,**ov,**d['metrics'],'wall_seconds':d['wall_seconds'],'result_path':str((out/'result.json').relative_to(ROOT))}
def main():
 rows=[]
 for spec in RUNS:
  result_file=RROOT/spec[0]/'result.json'
  if result_file.exists():
   d=json.loads(result_file.read_text());rows.append({'run_id':spec[0],'n':spec[1],**spec[2],**d['metrics'],'wall_seconds':d['wall_seconds'],'result_path':str(result_file.relative_to(ROOT))})
  else: rows.append(run(*spec))
 best=max(rows,key=lambda r:(r['rephrase_image_acc'],r['gen_avg']))
 ov={k:best[k] for k in ('lar_random_start','lar_pgd_steps','lar_step_size','lar_joint_perturbation','pmrl_spectral_alignment','pmrl_regularization_weight')}
 if 'lap_epsilon' in best: ov['lap_epsilon']=best['lap_epsilon']
 rows.append(run('paper-confirm-'+best['run_id'],100,ov));p=ROOT/'results/QWEN_WISE_IC_PAPER_GENM_SUMMARY.json';p.write_text(json.dumps(rows,indent=2));print(json.dumps({'best':best['run_id'],'confirm':rows[-1],'summary':str(p)}))
if __name__=='__main__':main()
