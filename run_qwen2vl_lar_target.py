#!/usr/bin/env python3
"""Screen explicit target supervision on paper-style LAR variants."""
import json,os,subprocess,time
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parent;BASE=yaml.safe_load((ROOT/'hparams/WISE/qwen2vl_ic_lap_pmrl.yaml').read_text())
RUNS=[
 ('lar-target-w0p05',20,0.05),('lar-target-w0p1',20,0.1),('lar-target-w0p25',20,0.25),('lar-target-w0p5',20,0.5)]
C=ROOT/'hparams/WISE/qwen2vl_lar_target';R=ROOT/'results/QWEN_WISE_IC_LAR_TARGET_RUNS';L=ROOT/'run_logs/wise_ic_qwen2vl_lar_target'
def run(rid,n,w):
 cfg=dict(BASE);cfg.update({'lar_random_start':True,'lar_pgd_steps':1,'lar_step_size':0.001,'lap_epsilon':0.001,'lar_joint_perturbation':True,'pmrl_spectral_alignment':False,'pmrl_regularization_weight':0.5,'lar_target_loss_weight':w});C.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True);L.mkdir(parents=True,exist_ok=True)
 cp=C/f'{rid}.yaml';cp.write_text(yaml.safe_dump(cfg,sort_keys=False));out=R/rid;ld=L/rid;out.mkdir(parents=True,exist_ok=True);ld.mkdir(parents=True,exist_ok=True)
 env=os.environ.copy();env.update({'OMP_NUM_THREADS':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONPATH':str(ROOT),'PMRL_TEST_SIZE':str(n),'PMRL_SEED':'42','PMRL_WISE_HPARAMS':str(cp.relative_to(ROOT)),'PMRL_OUTPUT_DIR':str(out.relative_to(ROOT))});start=time.time()
 with (ld/'run.log').open('w') as f:p=subprocess.run(['/root/miniconda3/envs/easyedit/bin/python','run_wise_qwen2vl_ic.py'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
 (ld/'status.txt').write_text(f'WALL_SECONDS={time.time()-start}\nEXIT_CODE={p.returncode}\n');assert p.returncode==0,rid;d=json.loads((out/'result.json').read_text());return {'run_id':rid,'n':n,'lar_target_loss_weight':w,**d['metrics'],'wall_seconds':d['wall_seconds'],'result_path':str((out/'result.json').relative_to(ROOT))}
def main():
 rows=[run(*x) for x in RUNS];best=max(rows,key=lambda x:(x['rephrase_image_acc'],x['gen_avg']));rows.append(run('lar-target-confirm-'+best['run_id'],100,best['lar_target_loss_weight']));p=ROOT/'results/QWEN_WISE_IC_LAR_TARGET_SUMMARY.json';p.write_text(json.dumps(rows,indent=2));print(json.dumps({'best':best['run_id'],'confirm':rows[-1],'summary':str(p)}))
if __name__=='__main__':main()
