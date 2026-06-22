#!/bin/bash
python ./examples/RLAD.py --env halfcheetah-medium-v2        --ad_module fanogan --all_saves saves_fanogan --weight_function sigmoid_theoretically && \
python ./examples/RLAD.py --env halfcheetah-medium-expert-v2  --ad_module fanogan --all_saves saves_fanogan --weight_function sigmoid_theoretically && \
python ./examples/RLAD.py --env halfcheetah-medium-replay-v2  --ad_module fanogan --all_saves saves_fanogan --weight_function sigmoid_theoretically && \
python ./examples/RLAD.py --env halfcheetah-random-v2         --ad_module fanogan --all_saves saves_fanogan --weight_function sigmoid_theoretically
