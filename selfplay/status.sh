#!/bin/bash
# One line per run: done (final checkpoint saved), running, dead, or waiting (not started yet).
# Give run names to check other runs:   bash status.sh novelty-0 prediction_error-1
# With no names it checks the eight reproduction runs.
cd ~/block-world/selfplay
names=("$@")
if [ ${#names[@]} -eq 0 ]; then
  names=(self-play-0 uniform-0 shuffle-0 loss_delta-0 self-play-1 uniform-1 shuffle-1 loss_delta-1)
fi
for name in "${names[@]}"; do
  last=2048; [ "${name%-*}" = uniform ] && last=3072
  if [ -f "results/$name/learner_$(printf %04d $last).pth" ]; then state=done
  elif pgrep -f "python train.py results/$name " > /dev/null; then state=running
  elif [ ! -e "results/$name.log" ]; then state=waiting
  else state=dead; fi
  echo "$name $state $(ls results/$name 2>/dev/null | grep -o "learner_[0-9]*" | sort | tail -1)"
done
