#!/bin/bash
# usage: up2.sh SESSION -- make folders, then upload code/ and p2/ (20 MB parts), 4 tries each
S=$1; cd /tmp/kf
timeout 120 colab exec -s $S -f mk2.py > /dev/null 2>&1
for f in code/* p2/*; do
  ok=0
  for t in 1 2 3 4; do timeout 300 colab upload -s $S $f /content/$f 2>&1 | grep -q Uploaded && { ok=1; break; }; sleep 3; done
  [ $ok = 1 ] && echo "ok $f" || echo "FAILED $f"
done
echo UPLOAD DONE
