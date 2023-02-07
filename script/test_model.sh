dataset=$1
exp_name=$2
testset=$3

nohup python -u test_model.py --dataset $dataset \
            --exp_name ${exp_name} \
            --batch_size 768 --regen \
            --testset $testset > logs/test_${dataset}_${exp_name}_${testset}.log &