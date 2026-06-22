#!/bin/bash
python ./examples/RLAD.py --env hopper-medium-v2        --ad_module maf --all_saves saves_maf --weight_function identity && \
python ./examples/RLAD.py --env hopper-medium-expert-v2  --ad_module maf --all_saves saves_maf --weight_function identity && \
python ./examples/RLAD.py --env hopper-medium-replay-v2  --ad_module maf --all_saves saves_maf --weight_function identity && \
python ./examples/RLAD.py --env hopper-random-v2         --ad_module maf --all_saves saves_maf --weight_function identity
