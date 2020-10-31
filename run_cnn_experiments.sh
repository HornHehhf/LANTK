#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python cnn.py neg=3 pos=5 > cnn_3_5.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python cnn.py neg=2 pos=4 > cnn_2_4.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python cnn.py neg=2 pos=3 > cnn_2_3.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python cnn.py neg=1 pos=9 > cnn_1_9.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 python cnn.py neg=2 pos=5 > cnn_2_5.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python cnn.py neg=4 pos=5 > cnn_4_5.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 python cnn.py neg=3 pos=4 > cnn_3_4.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python cnn.py neg=3 pos=6 > cnn_3_6.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python cnn.py neg=4 pos=6 > cnn_4_6.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python cnn.py neg=2 pos=6 > cnn_2_6.log 2>&1 &