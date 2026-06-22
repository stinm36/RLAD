#!/bin/bash
python ./examples/RLAD_CQL.py --env hopper-medium-v2        --ad_module svdd --all_saves saves_cql_ad --weight_function identity && \
python ./examples/RLAD_CQL.py --env hopper-medium-expert-v2  --ad_module svdd --all_saves saves_cql_ad --weight_function identity && \
python ./examples/RLAD_CQL.py --env hopper-medium-replay-v2  --ad_module svdd --all_saves saves_cql_ad --weight_function identity && \
python ./examples/RLAD_CQL.py --env hopper-random-v2         --ad_module svdd --all_saves saves_cql_ad --weight_function identity
