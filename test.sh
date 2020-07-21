#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python mae.py --num_filters 192 192 192 --filters_offset 0 0 0 --model_ID 0 --condition 128 --condition_norm 2048.0 --checkpoint_dir /models/mae/ --inputPath /dataset/ --evaluation_name mae evaluate



