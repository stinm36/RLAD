#!/bin/bash
python ./examples/RLAD.py --env antmaze-umaze-v0          --ad_module svdd --all_saves saves_antmaze --weight_function identity --ad_train True && \
python ./examples/RLAD.py --env antmaze-umaze-diverse-v0   --ad_module svdd --all_saves saves_antmaze --weight_function identity --ad_train True && \
python ./examples/RLAD.py --env antmaze-medium-diverse-v0  --ad_module svdd --all_saves saves_antmaze --weight_function identity --ad_train True
