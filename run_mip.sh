#! /bin/bash

model_name_or_path='/workspace/models/llama/llama-7B/'

train_file='data/train.jsonl'
validation_file='data/eval.jsonl'
output_dir=saved_mip_epoch4
output_dir_lora=saved_models_lora
mkdir -p ${output_dir}
#mkdir -p ${output_dir_lora}

cache_dir=hf_ptmodel_cache_dir
mkdir -p ${cache_dir}
cutoff_len=400


# ft
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' torchrun --nproc_per_node 8 train_mip.py \
    --model_name_or_path ${model_name_or_path} \
    --prompt_template 'qa' \
    --deepspeed configs/deepspeed_config_stage3.json \
    --train_file ${train_file} \
    --validation_file ${validation_file} \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 4 \
    --model_max_length ${cutoff_len} \
    --save_strategy "steps" \
    --save_total_limit 3 \
    --learning_rate 1e-4 \
    --weight_decay 0.00001 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --evaluation_strategy "steps" \
    --fp16 True \
    --seed 1234 \
    --gradient_checkpointing True \
    --cache_dir ${cache_dir} \
    --output_dir ${output_dir} \
    --do_train \
    --do_eval \
    --pt_train_file_dir ./pt_kefu\
    --pt_validation_file_dir ./pt_kefu \
    --model_type llama \
    --max_train_samples -1 \
    --max_eval_samples -1 
