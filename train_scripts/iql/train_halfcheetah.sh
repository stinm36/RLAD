#!/bin/bash
python ./examples/RL_IQL.py --env halfcheetah-medium-v2        --all_saves saves_iql && \
python ./examples/RL_IQL.py --env halfcheetah-medium-expert-v2  --all_saves saves_iql && \
python ./examples/RL_IQL.py --env halfcheetah-medium-replay-v2  --all_saves saves_iql && \
python ./examples/RL_IQL.py --env halfcheetah-random-v2         --all_saves saves_iql
