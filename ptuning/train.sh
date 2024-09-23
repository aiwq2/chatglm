PRE_SEQ_LEN=128
LR=2e-2

CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --do_train \
    --train_file AIOPS/jianjie5/train_jianjie5_final_rewrite.json \
    --validation_file AIOPS/jianjie5/dev_jianjie5_final_rewrite.json \
    --prompt_column content \
    --response_column summary \
    --overwrite_cache \
    --model_name_or_path THUDM/chatglm-6b \
    --output_dir output/aiops-chatglm-6b-pt-$PRE_SEQ_LEN-$LR-jianjie5_0921 \
    --overwrite_output_dir \
    --max_source_length 800 \
    --max_target_length 64 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --predict_with_generate \
    --max_steps 500 \
    --logging_steps 10 \
    --save_steps 500 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4

