#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
nohup python -m src.main --name rankclip_cc3m_v6 \
    --batch_size 512 \
    --train_data /home/user/data/zym/codespace/data/data/cc3m/cleaned_train.csv \
    --validation_data /home/user/data/zym/codespace/data/data/cc3m/cleaned_val.csv \
    --image_key image \
    --caption_key caption \
    --cyrankclip \
    --wandb \
    --device gpu &