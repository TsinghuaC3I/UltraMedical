#!/bin/sh

export now_time="$(date -u -d '+8 hours' '+%Y-%m%d-%H%M')"
echo ${now_time}

# --num_gpus $MLP_WORKER_GPU  \
# --num_nodes $MLP_WORKER_NUM \
# --hostfile=$MLP_MPI_HOSTFILE \
# --master_addr $MLP_WORKER_0_HOST \
# --master_port=$MLP_WORKER_0_PORT \

for lr in 2e-5 5e-6 1e-5 2e-5
do
        deepspeed \
            --num_gpus 8 \
            src/finetune/run_sft.py src/finetune/config/sft_config_full.yaml \
            --learning_rate=${lr} \
            --num_train_epochs=3 \
            --output_dir="outputs/ultramedical/Meta-Llama-3.1-8B-Instruct-${lr}" \
            1>logs/sft_full_8b_ultramedical_${now_time}.log 2>&1
done