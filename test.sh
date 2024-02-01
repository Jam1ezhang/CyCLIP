#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
python -m src.main \
    --name cyclip11_eval_imagenet_1k \
    --eval_data_type ImageNet1K \
    --eval_test_data_dir /home/user/data/zym/codespace/data/imagenet_val \
    --checkpoint /home/user/data/zym/codespace/CyCLIP/logs/cyclip_cc3m/checkpoints/epoch_11.pt \
    --wandb