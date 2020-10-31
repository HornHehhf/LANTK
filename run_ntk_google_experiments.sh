#!/usr/bin/env bash
nohup python ntk_google.py neg=3 pos=4 > ntk_google_3_4.log 2>&1 &

nohup python ntk_google.py neg=3 pos=6 > ntk_google_3_6.log 2>&1 &

nohup python ntk_google.py neg=4 pos=6 > ntk_google_4_6.log 2>&1 &

nohup python ntk_google.py neg=2 pos=6 > ntk_google_2_6.log 2>&1 &
