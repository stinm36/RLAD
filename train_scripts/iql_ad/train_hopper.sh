#!/bin/bash
python ./examples/RLAD_IQL.py --env hopper-medium-v2        --ad_module svdd --all_saves saves_iql_ad --weight_function identity && \
python ./examples/RLAD_IQL.py --env hopper-medium-expert-v2  --ad_module svdd --all_saves saves_iql_ad --weight_function identity && \
python ./examples/RLAD_IQL.py --env hopper-medium-replay-v2  --ad_module svdd --all_saves saves_iql_ad --weight_function identity && \
python ./examples/RLAD_IQL.py --env hopper-random-v2         --ad_module svdd --all_saves saves_iql_ad --weight_function identity
