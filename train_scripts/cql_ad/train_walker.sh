#!/bin/bash
python ./examples/RLAD_CQL.py --env walker2d-medium-v2        --ad_module svdd --all_saves saves_cql_ad --weight_function identity && \
python ./examples/RLAD_CQL.py --env walker2d-medium-expert-v2  --ad_module svdd --all_saves saves_cql_ad --weight_function identity && \
python ./examples/RLAD_CQL.py --env walker2d-medium-replay-v2  --ad_module svdd --all_saves saves_cql_ad --weight_function identity && \
python ./examples/RLAD_CQL.py --env walker2d-random-v2         --ad_module svdd --all_saves saves_cql_ad --weight_function identity
