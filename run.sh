export NCCL_DEBUG=WARN
export FI_PROVIDER=tcp
ACCELERATE_CONFIG="configs/default_config.yaml"
PORT=$(( ((RANDOM<<15)|RANDOM) % 27001 + 2000 ))
echo $PORT

# "yandex/stable-diffusion-3.5-large-alchemist"
# "stabilityai/stable-diffusion-3.5-large"
MODEL_NAME="stabilityai/stable-diffusion-3.5-large"
#DATASET_PATH="configs/data/mj_sd3.5_cfg4.5_40_steps_preprocessed.yaml"
DATASET_PATH="/mnt/remotes/sensei-fs/users/nstarodubcev/projects/refining_dataset/general_refiner/dataset_alchemist"
#42

accelerate launch --num_processes=8 --multi_gpu --mixed_precision fp16 --main_process_port $PORT main.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --train_data_dir=$DATASET_PATH \
    --current_task="refiner_sd3" \
    --output_dir="results" \
    --seed=42 \
    --train_batch_size=4 \
    --eval_batch_size=4 \
    --learning_rate=6e-5 \
    --learning_rate_cls=3e-4 \
    --lr_scheduler="constant_with_warmup" \
    --lr_scheduler_cls="constant" \
    --lr_warmup_steps=300 \
    --refining_timestep_index=25 \
    --refining_scale=1.0 \
    --do_gan \
    --num_steps_fake_dmd=2 \
    --num_discriminator_layers=4 \
    --cls_blocks=20 \
    --pdm_blocks=20 \
    --cfg_teacher=3.5 \
    --cfg_fake=0.0 \
    --gen_cls_loss_weight=1e-2 \
    --guidance_cls_loss_weight=1e-2 \
    --rank=128 \
    --apply_lora_to_attn_projections \
    --apply_lora_to_mlp_projections \
    --validation_steps=50 \
    --evaluation_steps=10000 \
    --max_train_steps=1000 \
    --checkpointing_steps=5000 \
    --max_eval_samples=5000 \
    --resume_from_checkpoint=latest \
    --gradient_checkpointing \
    --text_column="text" \
    --image_column="image" \
    --coco_ref_stats_path stats/fid_stats_mscoco256_val.npz \
    --inception_path stats/pt_inception-2015-12-05-6726825d.pth \
    --pickscore_model_name_or_path yuvalkirstain/PickScore_v1 \
    --clip_model_name_or_path laion/CLIP-ViT-H-14-laion2B-s32B-b79K \
    --report_to="tensorboard" \
    # --offload_text_encoders \
    # --text_embedding_column="vit_l_14_text_embedding" \
    # --text_embedding_2_column="vit_bigg_14_text_embedding" \
    # --text_embedding_3_column="t5xxl_text_embedding" \
    # --pooled_text_embedding_column="vit_l_14_pooled_text_embedding" \
    # --pooled_text_embedding_2_column="vit_bigg_14_pooled_text_embedding" \
