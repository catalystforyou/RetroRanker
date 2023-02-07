
dataset=${dataset:-vanilat}
exp_name=${exp_name:-1101}

nohup python -u train_model.py --dataset ${dataset} --exp_name ${exp_name} -nw 6 > logs/train_${dataset}_${exp_name}.log &
