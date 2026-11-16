# Colab VM: join the uploaded parts, then run the resize test. Blocking on purpose (an idle kernel gets reclaimed).
# Args via --env: SIZE (size_knn npz name in /content/f3dr/data), STEP. Log: /content/f3dr/run.log
import glob
import os
import subprocess

D = "/content/f3dr/data"
for name in ("kf_v27_e22.npz", "gt_oof.npz"):
    parts = sorted(glob.glob(f"{D}/{name}.part*"))
    if parts and not os.path.exists(f"{D}/{name}"):
        with open(f"{D}/{name}", "wb") as out:
            for p in parts:
                with open(p, "rb") as f:
                    out.write(f.read())
    print(name, os.path.getsize(f"{D}/{name}"), flush=True)
cmd = ["python", "/content/f3dr/baselines/tools/size_rescale_oof.py", f"{D}/kf_v27_e22.npz", f"{D}/{os.environ['SIZE']}",
       os.environ["STEP"], "/content/f3dr/rescale.npz", f"{D}/gt_oof.npz"]
with open("/content/f3dr/run.log", "w") as log:
    r = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
print("EXIT", r.returncode)
