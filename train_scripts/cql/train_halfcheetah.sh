#!/bin/bash
python ./examples/RL_CQL.py --env halfcheetah-medium-v2        --all_saves saves_cql && \
python ./examples/RL_CQL.py --env halfcheetah-medium-expert-v2  --all_saves saves_cql && \
python ./examples/RL_CQL.py --env halfcheetah-medium-replay-v2  --all_saves saves_cql && \
python ./examples/RL_CQL.py --env halfcheetah-random-v2         --all_saves saves_cql
