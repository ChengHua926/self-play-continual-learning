#!/bin/bash
# Stops the robot runs once a round has been scored and saved: record.json, written last at each
# save, then holds that round's test 1 scores. Run on the GPU machine, so the stop does not depend
# on anything else staying up:   setsid -f bash stop_at.sh 1024 > runs/stop_at.log 2>&1 < /dev/null
cd ~/block-world/robot
until ../.venv/bin/python -c "import json, sys; sys.exit(0 if '$1' in json.load(open('runs/record.json'))[0]['scores'] else 1)" 2> /dev/null; do
  sleep 30
done
pkill -f "self_play.py runs "
echo "stopped after round $1 was saved, at $(date -u)"
