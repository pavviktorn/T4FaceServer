#!/bin/bash
set -e  # stop the script if any command fails

#conda init
#conda activate torch260_cu124

# Shared PYTHONPATH
export PYTHONPATH=/datasets/work/vLLM/FACE_FEATS/FFAA-feats:$PYTHONPATH

############################################
# GLOBAL SETTINGS
############################################

# Device map variable (EDIT HERE if needed)
DEVICE_MAP="localhost:0,1,2,3"

# Base_vLLM_path="/datasets/work/vLLM/FFAA-master/checkpoints/ffaa-mistral-7b"
Base_vLLM_path="/datasets/work/vLLM/FACE_FEATS/FFAA-feats/checkpoints_4+13fmt_fix/effaa-llava-mistral-7b-lora_3"
Finetuned_vLLM_path="/datasets/work/vLLM/FACE_FEATS/FFAA-feats/checkpoints_4+13fmt_fix/effaa-llava-mistral-7b-lora"
Merged_vLLM_path="/datasets/work/vLLM/FACE_FEATS/FFAA-feats/checkpoints_4+13fmt_fix/effaa-llava-mistral-7b-lora_4"
all_json="/datasets/newout/vqa_info_2+13+4+3_fmt/eFFAA_feats.json"
base_json="/datasets/newout/vqa_info_2+13+4+3_fmt/eFFAA_feats.json"
eval_json="/datasets/newout/vqa_info_2+13+4+3_fmt/eFFAA_feats_eval.json"

############################################
# STEP 1 — Finetune Mistral LoRA
############################################

echo "==== Step 1: Finetune Mistral LoRA ===="

deepspeed --master_port 25642 --include $DEVICE_MAP \
    llava/train/train_mem.py \
    --lora_enable True --lora_r 32 --lora_alpha 48 --lora_dropout 0.05 --mm_projector_lr 1e-6 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path $Base_vLLM_path \
    --version v1 \
    --data_path $all_json \
    --image_folder /datasets/newout \
    --vision_tower ./models/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir $Finetuned_vLLM_path \
    --num_train_epochs 3 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 24 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to "none" \
    --eval_data_path $eval_json \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --metric_for_best_model eval_loss \
    --greater_is_better False


python3.12 merge_lora_weights.py \
--model-path $Finetuned_vLLM_path \
--model-base $Base_vLLM_path \
--save-model-path $Merged_vLLM_path

echo "==== ALL STEPS COMPLETED SUCCESSFULLY ===="
