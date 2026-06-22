#!/bin/bash
python ./examples/RLAD.py --env walker2d-medium-v2        --ad_module mahal --all_saves saves_mahal --weight_function sigmoid && \
python ./examples/RLAD.py --env walker2d-medium-expert-v2  --ad_module mahal --all_saves saves_mahal --weight_function sigmoid && \
python ./examples/RLAD.py --env walker2d-medium-replay-v2  --ad_module mahal --all_saves saves_mahal --weight_function sigmoid && \
python ./examples/RLAD.py --env walker2d-random-v2         --ad_module mahal --all_saves saves_mahal --weight_function sigmoid
