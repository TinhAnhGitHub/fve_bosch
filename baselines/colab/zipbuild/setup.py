# Colab L4 VM: submission zips from checkpoints (tools/mix_zip.py). Blocking on purpose.
# Env: HF_TOKEN (DINOv3 weights), BUILDS = "name=member member ...;name=..." with checkpoint file names in /content/ck
# (a member may pool several, joined by commas). Photos: /content/up/photos.tar.part* (pack_photos.py).
# Output /content/zips/<name>.zip, log /content/run.log (ends with "ZIPS DONE"). Finished zips are skipped on rerun.
import glob
import os
import subprocess
import tarfile

F = "/content/f3dr"
os.makedirs("/content/zips", exist_ok=True)
if not os.path.isdir(f"{F}/data/f3d/eval"):
    with open("/content/up/photos.tar", "wb") as out:
        for p in sorted(glob.glob("/content/up/photos.tar.part*")):
            with open(p, "rb") as f:
                out.write(f.read())
    with tarfile.open("/content/up/photos.tar") as tar:
        tar.extractall(f"{F}/data/f3d")
subprocess.run("pip install -q trimesh", shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
with open("/content/run.log", "a") as log:
    for build in os.environ["BUILDS"].split(";"):
        name, members = build.split("=")
        out = f"/content/zips/{name}.zip"
        if os.path.exists(out):
            continue
        args = [",".join(f"/content/ck/{c}" for c in m.split(",")) for m in members.split()]
        r = subprocess.run(["python", f"{F}/baselines/tools/mix_zip.py", out + ".tmp", *args], cwd=F,
                           stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        if r.returncode == 0:
            os.replace(out + ".tmp", out)
        log.write(f"BUILD {name} EXIT {r.returncode}\n")
        log.flush()
    log.write("ZIPS DONE\n")
print("ZIPS DONE")
