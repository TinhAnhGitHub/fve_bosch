# Colab T4 VM: join the uploaded parts, then run tools/mix_oof.py on the OOF files named in env FILES (in order).
# Blocking on purpose. Log /content/f3dr/run.log, result /content/f3dr/mix_oof.npz
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
cmd = ["python", "/content/f3dr/baselines/tools/mix_oof.py", f"{D}/gt_oof.npz", "/content/f3dr/mix_oof.npz",
       *[f"{D}/{f}" for f in os.environ["FILES"].split(",")]]
with open("/content/f3dr/run.log", "w") as log:
    r = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
print("EXIT", r.returncode)
