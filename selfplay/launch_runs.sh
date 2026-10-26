#!/bin/bash
# Start 99k runs in the background on one GPU.
# Run it from the selfplay folder:   bash launch_runs.sh [name ...]
# Names are <arm>-<seed>, e.g. self-play-0 or uniform-1. With no names it starts all eight
# (four arms x seeds 0 and 1). Each run saves into results/<name>/ and logs to results/<name>.log.
# A run that is started again continues from its last saved state (every 256 rounds).
#
# The uniform arm uses 1,024 programs per round, as the authors' uniform baseline did at this size
# (their figures/fig2 data, prior_programs_per_round.json); self-play uses 1,536 at every size. So
# uniform runs to round 3072, where it has used as much compute as self-play at round 2048.
#
# PYTHON sets the interpreter (default: the venv one level up, as on the Mac).
PYTHON=${PYTHON:-../.venv/bin/python}
names=("$@")
if [ ${#names[@]} -eq 0 ]; then
  names=(self-play-0 uniform-0 shuffle-0 loss_delta-0 self-play-1 uniform-1 shuffle-1 loss_delta-1)
fi
mkdir -p results
for name in "${names[@]}"; do
  arm=${name%-*}
  seed=${name##*-}
  settings="arm=$arm seed=$seed"
  if [ "$arm" = uniform ]; then
    settings="$settings fresh=1024 mutants=0 replay=0 rounds=3073"
  fi
  # 3 CPU threads each for the machine, so several runs share the CPUs without fighting
  NUMBA_NUM_THREADS=3 setsid -f bash -c \
    "$PYTHON train.py results/$name $settings > results/$name.log 2>&1 < /dev/null" \
    > /dev/null 2>&1 < /dev/null
  echo "started $name ($settings)"
done
