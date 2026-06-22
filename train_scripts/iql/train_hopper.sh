#!/bin/bash
python ./examples/RL_IQL.py --env hopper-medium-v2        --all_saves saves_iql && \
python ./examples/RL_IQL.py --env hopper-medium-expert-v2  --all_saves saves_iql && \
python ./examples/RL_IQL.py --env hopper-medium-replay-v2  --all_saves saves_iql && \
python ./examples/RL_IQL.py --env hopper-random-v2         --all_saves saves_iql
