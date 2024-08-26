#!/bin/sh

export now_time=$(date -u -d '+8 hours' '+%Y-%m%d-%H%M')
echo ${now_time}

# --num_gpus $MLP_WORKER_GPU  \
# --num_nodes $MLP_WORKER_NUM \
# --hostfile=$MLP_MPI_HOSTFILE \
# --master_addr $MLP_WORKER_0_HOST \
# --master_port=$MLP_WORKER_0_PORT \

for lr in  1e-5
do
    echo ${lr}
    deepspeed \
        --num_gpus 8 \
        src/finetune/run_dpo.py src/finetune/config/dpo_config_full.yaml \
        --model_name_or_path="outputs/ultramedical/Meta-Llama-3.1-8B/1e-5-length8192" \
        --save_strategy="steps" \
        --num_train_epochs=1 \
        --output_dir="outputs/ultramedical/Meta-Llama-3.1-8B-Instruct-1e-5-dpo/${lr}" \
        1>logs/dpo_8b_${now_time}.log 2>&1
done