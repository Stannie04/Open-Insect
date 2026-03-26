#!/bin/bash

export HF_TOKEN="your_huggingface_token_here"

python download.py --download_dir . --resize_size 224 --region_name "c-america"
python download.py --download_dir . --resize_size 224 --region_name "ne-america"
python download.py --download_dir . --resize_size 224 --region_name "w-europe"

python download_pretrained_weights.py --weight_dir ./model_checkpoints