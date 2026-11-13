import os, subprocess
# RUNS = "key:fold,key:fold,...". Runs go in waves of WAVE at a time (RAM: ~5 GB per run), one run_exps call per wave.
R = {"v26": "--arch talk --right-handed --seed 2 --epochs 32",
     "tds": "--arch talk --deep-sup 0.2 --epochs 40",
     "ds": "--arch sheet --deep-sup 0.2 --epochs 40",
     "r1": "--arch sheet --drop-path 0.15 --attn-drop 0.1 --grid-jitter 0.5 --seed 1 --epochs 40",
     "nocat": "--arch talk --no-cat --epochs 40"}
F = "--encoder dinov3l --folds baselines/folds/f3dr_k5_v1.csv"
runs = []
for item in os.environ["RUNS"].split(","):
    key, k = item.split(":")
    last = R[key].split("--epochs ")[1]
    runs += [f"kf_{key}_f{k}", f"{R[key]} {F} --fold {k} --save-epochs 12:{last}"]
wave = int(os.environ.get("WAVE", "5")) * 2
calls = [" ".join(["python code/run_exps.py /content/run /content/data /content/code"] + [f'"{a}"' for a in runs[i : i + wave]])
         for i in range(0, len(runs), wave)]
script = "cd /content && (" + " && ".join(calls) + ") > /content/run.log 2>&1"
subprocess.Popen(["bash", "-c", script], start_new_session=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print("LAUNCHED", runs[::2], "waves of", wave // 2)
