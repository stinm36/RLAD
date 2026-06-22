#!/bin/bash
python ./examples/RLAD.py --env hopper-medium-v2        --ad_module mahal --all_saves saves_mahal --weight_function sigmoid && \
python ./examples/RLAD.py --env hopper-medium-expert-v2  --ad_module mahal --all_saves saves_mahal --weight_function sigmoid && \
python ./examples/RLAD.py --env hopper-medium-replay-v2  --ad_module mahal --all_saves saves_mahal --weight_function sigmoid && \
python ./examples/RLAD.py --env hopper-random-v2         --ad_module mahal --all_saves saves_mahal --weight_function sigmoid
