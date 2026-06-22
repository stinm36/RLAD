#!/bin/bash
python ./examples/RLAD.py --env halfcheetah-medium-v2        --ad_module dagmm --all_saves saves_dagmm --weight_function identity && \
python ./examples/RLAD.py --env halfcheetah-medium-expert-v2  --ad_module dagmm --all_saves saves_dagmm --weight_function identity && \
python ./examples/RLAD.py --env halfcheetah-medium-replay-v2  --ad_module dagmm --all_saves saves_dagmm --weight_function identity && \
python ./examples/RLAD.py --env halfcheetah-random-v2         --ad_module dagmm --all_saves saves_dagmm --weight_function identity
