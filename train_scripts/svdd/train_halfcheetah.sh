#!/bin/bash
python ./examples/RLAD.py --env halfcheetah-medium-v2        --ad_module svdd --all_saves saves_svdd --weight_function identity && \
python ./examples/RLAD.py --env halfcheetah-medium-expert-v2  --ad_module svdd --all_saves saves_svdd --weight_function identity && \
python ./examples/RLAD.py --env halfcheetah-medium-replay-v2  --ad_module svdd --all_saves saves_svdd --weight_function identity && \
python ./examples/RLAD.py --env halfcheetah-random-v2         --ad_module svdd --all_saves saves_svdd --weight_function identity
