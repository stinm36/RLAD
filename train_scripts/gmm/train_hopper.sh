#!/bin/bash
python ./examples/RLAD.py --env hopper-medium-v2        --ad_module gmm --all_saves saves_gmm --weight_function identity --gmm_k 2 && \
python ./examples/RLAD.py --env hopper-medium-expert-v2  --ad_module gmm --all_saves saves_gmm --weight_function identity --gmm_k 2 && \
python ./examples/RLAD.py --env hopper-medium-replay-v2  --ad_module gmm --all_saves saves_gmm --weight_function identity --gmm_k 2 && \
python ./examples/RLAD.py --env hopper-random-v2         --ad_module gmm --all_saves saves_gmm --weight_function identity --gmm_k 2
