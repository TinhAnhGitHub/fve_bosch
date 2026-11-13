#!/bin/bash
# usage: rebuild.sh SESSION FOLDS -- full setup of a fresh session, then launch the v27 ft + frozen runs for FOLDS
S=$1; FOLDS=$2; cd /tmp/kf
timeout 1500 colab exec -s $S --timeout 1400 --env KAGGLE_API_TOKEN="$(cat ~/.kaggle/access_token)" -f setup.py > setup_$S.log 2>&1 &
./up2.sh $S > up_$S.log 2>&1
wait
grep -q "SETUP DONE" setup_$S.log || { echo "SETUP FAILED $S"; exit 1; }
grep -q FAILED up_$S.log && { echo "UPLOAD FAILED $S"; grep FAILED up_$S.log; exit 1; }
timeout 900 colab exec -s $S --timeout 850 -f assemble2.py 2>&1 | grep -E "^photos|^dino |tokens vs|ASSEMBLE"
timeout 300 colab exec -s $S --timeout 250 --env FOLDS=$FOLDS -f launch.py 2>&1 | tail -1
