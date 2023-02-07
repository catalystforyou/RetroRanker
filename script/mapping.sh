#!/bin/bash
dataset=$1 # vanilla rsmiles

for i in 0 1 2 3 4 5 6 7 test
do
   nohup python mapping.py --chunk_id $i --dataset $dataset > logs/mapping_${dataset}_$i.log &
done