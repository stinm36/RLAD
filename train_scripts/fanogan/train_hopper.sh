#!/bin/bash
python ./examples/RLAD.py --env hopper-medium-v2        --ad_module fanogan --all_saves saves_fanogan --weight_function sigmoid_theoretically && \
python ./examples/RLAD.py --env hopper-medium-expert-v2  --ad_module fanogan --all_saves saves_fanogan --weight_function sigmoid_theoretically && \
python ./examples/RLAD.py --env hopper-medium-replay-v2  --ad_module fanogan --all_saves saves_fanogan --weight_function sigmoid_theoretically && \
python ./examples/RLAD.py --env hopper-random-v2         --ad_module fanogan --all_saves saves_fanogan --weight_function sigmoid_theoretically
