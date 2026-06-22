#!/bin/bash
python ./examples/RLAD.py --env walker2d-medium-v2        --ad_module gmm --all_saves saves_gmm --weight_function identity --gmm_k 2 && \
python ./examples/RLAD.py --env walker2d-medium-expert-v2  --ad_module gmm --all_saves saves_gmm --weight_function identity --gmm_k 2 && \
python ./examples/RLAD.py --env walker2d-medium-replay-v2  --ad_module gmm --all_saves saves_gmm --weight_function identity --gmm_k 2 && \
python ./examples/RLAD.py --env walker2d-random-v2         --ad_module gmm --all_saves saves_gmm --weight_function identity --gmm_k 2
