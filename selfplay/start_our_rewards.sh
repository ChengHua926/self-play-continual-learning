#!/bin/bash
# Starts the 4 runs of our two rewards once all eight reproduction runs have finished.
# It runs here on the GPU machine, so the hand-off does not depend on the Mac staying awake.
# Started with:
#   setsid -f bash start_our_rewards.sh > results/start_our_rewards.log 2>&1 < /dev/null
cd ~/block-world/selfplay
ours=(novelty-0 novelty-1 prediction_error-0 prediction_error-1)
for name in "${ours[@]}"; do
  if [ -e "results/$name.log" ]; then echo "$name was already started; doing nothing"; exit 1; fi
done
until [ "$(bash status.sh | grep -c " done ")" = 8 ] && ! pgrep -f "python train.py" > /dev/null; do
  sleep 60
done
echo "the eight runs finished by $(date -u)"
# the code with our two rewards (commit 27542bf); the other files are the same as here
cp ../selfplay_next/{train,reward,compare,test_selfplay}.py .
bash launch_runs.sh "${ours[@]}"
