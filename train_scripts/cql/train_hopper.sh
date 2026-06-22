#!/bin/bash
python ./examples/RL_CQL.py --env hopper-medium-v2        --all_saves saves_cql && \
python ./examples/RL_CQL.py --env hopper-medium-expert-v2  --all_saves saves_cql && \
python ./examples/RL_CQL.py --env hopper-medium-replay-v2  --all_saves saves_cql && \
python ./examples/RL_CQL.py --env hopper-random-v2         --all_saves saves_cql
