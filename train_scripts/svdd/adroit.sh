#!/bin/bash
python ./examples/RLAD.py --env pen-human-v1       --ad_module svdd --all_saves saves_adroit --weight_function sigmoid_theoretically --ad_train True && \
python ./examples/RLAD.py --env pen-cloned-v1      --ad_module svdd --all_saves saves_adroit --weight_function sigmoid_theoretically --ad_train True && \
python ./examples/RLAD.py --env hammer-human-v1    --ad_module svdd --all_saves saves_adroit --weight_function sigmoid_theoretically --ad_train True && \
python ./examples/RLAD.py --env hammer-cloned-v1   --ad_module svdd --all_saves saves_adroit --weight_function sigmoid_theoretically --ad_train True && \
python ./examples/RLAD.py --env door-human-v1      --ad_module svdd --all_saves saves_adroit --weight_function sigmoid_theoretically --ad_train True && \
python ./examples/RLAD.py --env door-cloned-v1     --ad_module svdd --all_saves saves_adroit --weight_function sigmoid_theoretically --ad_train True && \
python ./examples/RLAD.py --env relocate-human-v1  --ad_module svdd --all_saves saves_adroit --weight_function sigmoid_theoretically --ad_train True && \
python ./examples/RLAD.py --env relocate-cloned-v1 --ad_module svdd --all_saves saves_adroit --weight_function sigmoid_theoretically --ad_train True
