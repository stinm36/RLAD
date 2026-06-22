#!/bin/bash
python ./examples/RL_IQL.py --env walker2d-medium-v2        --all_saves saves_iql && \
python ./examples/RL_IQL.py --env walker2d-medium-expert-v2  --all_saves saves_iql && \
python ./examples/RL_IQL.py --env walker2d-medium-replay-v2  --all_saves saves_iql && \
python ./examples/RL_IQL.py --env walker2d-random-v2         --all_saves saves_iql
