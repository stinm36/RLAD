#!/bin/bash
python ./examples/RLAD.py --env walker2d-medium-v2        --ad_module svdd --all_saves saves_svdd --weight_function identity && \
python ./examples/RLAD.py --env walker2d-medium-expert-v2  --ad_module svdd --all_saves saves_svdd --weight_function identity && \
python ./examples/RLAD.py --env walker2d-medium-replay-v2  --ad_module svdd --all_saves saves_svdd --weight_function identity && \
python ./examples/RLAD.py --env walker2d-random-v2         --ad_module svdd --all_saves saves_svdd --weight_function identity
