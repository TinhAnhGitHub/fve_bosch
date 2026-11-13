import subprocess
F = "--folds baselines/folds/f3dr_k5_v1.csv"
V27 = "--encoder dinov3l --arch talk --pos-emb --right-handed --seed 1"
runs = ["sm_f0", f"{V27} {F} --fold 0 --epochs 1 --save-epochs 1:1",
        "sm_all", f"{V27} {F} --fold all --epochs 1 --save-epochs 1:1",
        "sm_ft", f"{V27} {F} --fold 0 --epochs 1 --save-epochs 1:1 --finetune 4"]
cmd = ["bash", "-c", 'cd /content && python code/run_exps.py /content/smoke /content/data /content/code "$@" > /content/smoke.log 2>&1', "_", *runs]
subprocess.Popen(cmd, start_new_session=True)
print("SMOKE STARTED")
