#!/bin/bash

traindata=${traindata:-R-SMILES}
testdata=${testdata:-R-SMILES}
outdir=output/${traindata}_on_${testdata}/
bz=${bz:-96}
totalchunks=${totalchunks:-30}

mkdir -p $outdir
for (( i=0;i<$totalchunks;i++ ))
do
   echo $i
   python -u graphormer_eval.py --out_dir ${outdir} --chunk_id $i \
            --path model/${traindata}/gh/checkpoint_best.pt \
            --batch-size ${bz} --dataset-name ${testdata} 
done