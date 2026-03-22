#!/bin/bash

port=${1:-7788}
python -m tensorboard.main --logdir $PWD/lightning_logs --host 0.0.0.0 --port $port --samples_per_plugin "audio=1000, images=1000"
