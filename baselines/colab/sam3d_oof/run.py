# Colab T4 VM: join the uploaded parts, then score SAM 3D at predicted per-axis size vs ours on all 2,042 objects
# (tools/sam3d_oof.py). Blocking on purpose (an idle kernel gets reclaimed). Log: /content/f3dr/run.log
import glob
import os
import subprocess

D = "/content/f3dr/data"
for name in sorted({os.path.basename(p).split(".part")[0] for p in glob.glob(f"{D}/*.part*")}):
    if not os.path.exists(f"{D}/{name}"):
        with open(f"{D}/{name}", "wb") as out:
            for p in sorted(glob.glob(f"{D}/{name}.part*")):
                with open(p, "rb") as f:
                    out.write(f.read())
    print(name, os.path.getsize(f"{D}/{name}"), flush=True)
cmd = ["python", "/content/f3dr/baselines/tools/sam3d_oof.py", f"{D}/kf_v27_e22.npz", f"{D}/gt_oof.npz",
       f"{D}/size_knn.npz", f"{D}/size_knn_axes.npz", "/content/f3dr/sam3d_oof.npz",
       *sorted(glob.glob(f"{D}/sam3d_train_s*.npz"))]
with open("/content/f3dr/run.log", "w") as log:
    r = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
print("EXIT", r.returncode)
