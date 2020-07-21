#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python mae.py -v --train_glob="/dataset/*.png" train --patchsize 240 --num_filters 192 192 192 --filters_offset 0 0 0 --lambda 128 512 2048 --condition_norm 2048.0 --checkpoint_dir /models/mae --last_step 1200000



