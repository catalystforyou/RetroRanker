#!/bin/bash

dataset=${dataset:-R-SMILES}
nw=${nw:-2}
bz=${bz:-24}
target=gh
arch=${arch:-graphranker_large}

if [ -d "/teamdrive/projects/" ]; then
    cd /teamdrive/projects/retrorank/RetroRanker
fi


outdir=model/${dataset}/${target}
tb_dir=${outdir}/tb
log_file=${outdir}/training.log
mkdir -p ${tb_dir}

python -u graphormer_train.py --dataset-name ${dataset} --num-workers $nw \
            --save-dir ${outdir} \
            --ddp-backend=c10d \
            --task graph_rank \
            --criterion Gl2_loss \
            --log-file ${log_file} \
            --arch $arch --num-classes 1 --attention-dropout 0.1 \
            --act-dropout 0.1 --dropout 0.0 --optimizer adam \
            --adam-betas \(0.9,\ 0.999\) --adam-eps 1e-8 --clip-norm 3.0 \
            --weight-decay 0.01 --lr-scheduler polynomial_decay --power 1 \
            --warmup-updates 0 --total-num-update 500000 --lr 1e-5 \
            --end-learning-rate 1e-9 --batch-size $bz --fp16 --fp16-init-scale 4 \
            --max-epoch 20000 \
            --required-batch-size-multiple 1 --patience 30 \
            --log-interval 200 --log-format simple --tensorboard-logdir ${tb_dir} \
            --best-checkpoint-metric acc --maximize-best-checkpoint-metric

# append the following if reset 
# --reset-dataloader --reset-lr-scheduler --reset-meters --reset-optimizer