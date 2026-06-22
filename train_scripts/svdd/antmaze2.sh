#!/bin/bash
python ./examples/RLAD.py --env antmaze-medium-play-v0  --ad_module svdd --all_saves saves_antmaze --weight_function identity --ad_train True && \
python ./examples/RLAD.py --env antmaze-large-diverse-v0 --ad_module svdd --all_saves saves_antmaze --weight_function identity --ad_train True && \
python ./examples/RLAD.py --env antmaze-large-play-v0    --ad_module svdd --all_saves saves_antmaze --weight_function identity --ad_train True
