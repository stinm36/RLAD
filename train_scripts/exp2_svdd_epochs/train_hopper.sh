#!/bin/bash
# Exp 2: Deep SVDD pretraining epoch sensitivity — hopper-medium-v2
# For each epoch count: train SVDD with that many epochs (unique tag per run),
# then run RLAD-SAC loading that checkpoint. 3 seeds each.
ENV=hopper-medium-v2

for svdd_epochs in 100 300 500 1000; do
    for seed in 0 1 2; do
        TAG=e${svdd_epochs}_s${seed}
        python ./examples/RLAD.py \
            --env $ENV \
            --seed $seed \
            --ad_module svdd \
            --ad_train True \
            --svdd_train_epochs $svdd_epochs \
            --svdd_tag $TAG \
            --weight_function identity \
            --weight_actor False \
            --all_saves saves_exp2_svdd_epochs \
            --trial_name ${TAG} \
            --nepochs 3000
    done
done
