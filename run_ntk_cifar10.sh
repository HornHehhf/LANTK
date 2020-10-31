#!/usr/bin/env bash
for (( i = 0; i <= 9; i++ ))
do
    for (( j = $i + 1; j <= 9; j++ ))
    do
       python ntk.py neg_label=$i pos_label=$j
    done
done