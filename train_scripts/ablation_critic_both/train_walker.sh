#!/bin/bash
# Exp 1: Critic-only vs Both weighting ablation — walker2d-medium-v2
ENV=walker2d-medium-v2

for seed in 0 1 2 3 4; do
    python ./examples/RLAD.py \
        --env $ENV \
        --seed $seed \
        --ad_module svdd \
        --ad_train True \
        --weight_function identity \
        --weight_actor False \
        --all_saves saves_ablation_critic_only \
        --trial_name seed_${seed} \
        --nepochs 3000 && \
    python ./examples/RLAD.py \
        --env $ENV \
        --seed $seed \
        --ad_module svdd \
        --ad_train True \
        --weight_function identity \
        --weight_actor True \
        --all_saves saves_ablation_both \
        --trial_name seed_${seed} \
        --nepochs 3000
done
