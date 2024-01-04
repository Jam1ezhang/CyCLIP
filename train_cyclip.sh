#!/bin/bash
export CUDA_VISIBLE_DEVICES=6
nohup python -m src.main --name cyclip_cc3m \
    --batch_size 512 \
    --train_data /home/user/data/zym/codespace/CyCLIP/data/cc3m/cleaned_train.csv \
    --validation_data /home/user/data/zym/codespace/CyCLIP/data/cc3m/cleaned_val.csv \
    --image_key image \
    --caption_key caption \
    --cylambda1 0.25 \
    --cylambda2 0.25 \
    --device gpu &
