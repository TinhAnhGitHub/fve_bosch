import subprocess
print(subprocess.run("ls /content/photos; ls /content/photos/* | head; du -sh /content/photos; ls /content/photos/* | wc -l", shell=True, capture_output=True, text=True).stdout)
