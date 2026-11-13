import os, subprocess
F = "--folds baselines/folds/f3dr_k5_v1.csv --save-epochs 12:32"
V27 = "--encoder dinov3l --arch talk --pos-emb --right-handed --seed 1 --epochs 32"
runs = []
for k in os.environ["FOLDS"].split(","):
    runs += [f"ft_v27_f{k}", f"{V27} {F} --fold {k} --finetune 4", f"kf_v27_f{k}", f"{V27} {F} --fold {k}"]
cmd = ["bash", "-c", 'cd /content && python code/run_exps.py /content/run /content/data /content/code "$@" > /content/run.log 2>&1', "_", *runs]
subprocess.Popen(cmd, start_new_session=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("LAUNCHED", runs[::2])
