#!/bin/bash
python ./examples/RLAD.py --env halfcheetah-medium-v2        --ad_module mahal --all_saves saves_mahal --weight_function sigmoid && \
python ./examples/RLAD.py --env halfcheetah-medium-expert-v2  --ad_module mahal --all_saves saves_mahal --weight_function sigmoid && \
python ./examples/RLAD.py --env halfcheetah-medium-replay-v2  --ad_module mahal --all_saves saves_mahal --weight_function sigmoid && \
python ./examples/RLAD.py --env halfcheetah-random-v2         --ad_module mahal --all_saves saves_mahal --weight_function sigmoid
