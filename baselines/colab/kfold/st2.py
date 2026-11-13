import glob, subprocess
print(open("/content/run.log").read()[-600:])
for f in sorted(glob.glob("/content/run/out/*.train.log")):
    print("==", f.split("/")[-1], "|", " || ".join(open(f).read().splitlines()[-2:])[-300:])
