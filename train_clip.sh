#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
nohup python -m src.main --name clip_cc3m \
    --batch_size 512 \
    --train_data /home/user/data/zym/codespace/CyCLIP/data/cc3m/cleaned_train.csv \
    --validation_data /home/user/data/zym/codespace/CyCLIP/data/cc3m/cleaned_val.csv \
    --image_key image \
    --caption_key caption \
    --device gpu &