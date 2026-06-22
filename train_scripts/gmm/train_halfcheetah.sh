#!/bin/bash
python ./examples/RLAD.py --env halfcheetah-medium-v2        --ad_module gmm --all_saves saves_gmm --weight_function identity --gmm_k 2 && \
python ./examples/RLAD.py --env halfcheetah-medium-expert-v2  --ad_module gmm --all_saves saves_gmm --weight_function identity --gmm_k 2 && \
python ./examples/RLAD.py --env halfcheetah-medium-replay-v2  --ad_module gmm --all_saves saves_gmm --weight_function identity --gmm_k 2 && \
python ./examples/RLAD.py --env halfcheetah-random-v2         --ad_module gmm --all_saves saves_gmm --weight_function identity --gmm_k 2
