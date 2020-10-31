#!/usr/bin/env bash
nohup python ntl_google.py neg=3 pos=5 > ntl_google_3_5.log 2>&1 &
nohup python ntl_google.py neg=2 pos=4 > ntl_google_2_4.log 2>&1 &
nohup python ntl_google.py neg=2 pos=3 > ntl_google_2_3.log 2>&1 &
nohup python ntl_google.py neg=1 pos=9 > ntl_google_1_9.log 2>&1 &
nohup python ntl_google.py neg=2 pos=5 > ntl_google_2_5.log 2>&1 &

nohup python ntl_google.py neg=4 pos=5 > ntl_google_4_5_V2.log 2>&1 &
nohup python ntl_google.py neg=3 pos=4 > ntl_google_3_4_V2.log 2>&1 &
nohup python ntl_google.py neg=3 pos=6 > ntl_google_3_6_V2.log 2>&1 &
nohup python ntl_google.py neg=4 pos=6 > ntl_google_4_6_V2.log 2>&1 &
nohup python ntl_google.py neg=2 pos=6 > ntl_google_2_6_V2.log 2>&1 &