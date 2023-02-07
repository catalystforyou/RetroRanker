#!/bin/bash

ROOT=./
ProcessedDataDir=$ROOT/data
dataset=$1
GraphDataDir=$ROOT/data/$dataset/3_gendata

chunk_id=$2
total_chunks=$3
file_identifier=$4

cd $ROOT/RetroRanker
/opt/conda/bin/python generate_graphs.py --dataset $dataset \
    --chunk_id ${chunk_id} --total_chunks ${total_chunks} \
    --file_identifier ${file_identifier} --save_type pyg
