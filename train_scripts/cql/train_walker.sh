#!/bin/bash
python ./examples/RL_CQL.py --env walker2d-medium-v2        --all_saves saves_cql && \
python ./examples/RL_CQL.py --env walker2d-medium-expert-v2  --all_saves saves_cql && \
python ./examples/RL_CQL.py --env walker2d-medium-replay-v2  --all_saves saves_cql && \
python ./examples/RL_CQL.py --env walker2d-random-v2         --all_saves saves_cql
