#!/bin/bash

nohup python -m src.main --name clip_cc3m --train_data /home/user/data/zym/codespace/CyCLIP/data/cc3m/train.csv --validation_data /home/user/data/zym/codespace/CyCLIP/data/cc3m/val.csv --image_key i --caption_key captions --device gpu --device_ids 7 &