#!/bin/bash
python ./examples/RLAD_CQL.py --env halfcheetah-medium-v2        --ad_module svdd --all_saves saves_cql_ad --weight_function identity && \
python ./examples/RLAD_CQL.py --env halfcheetah-medium-expert-v2  --ad_module svdd --all_saves saves_cql_ad --weight_function identity && \
python ./examples/RLAD_CQL.py --env halfcheetah-medium-replay-v2  --ad_module svdd --all_saves saves_cql_ad --weight_function identity && \
python ./examples/RLAD_CQL.py --env halfcheetah-random-v2         --ad_module svdd --all_saves saves_cql_ad --weight_function identity
