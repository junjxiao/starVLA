#!/bin/bash
# DEEPSPEED_CONFIG_FILE=starVLA/config/deepseeds/zero0.json,
ENVS="CHECKPOINT_BASEDIR=/mnt/workspace/zengshuang.zs/checkpoints,WANDB_MODE=offline,HF_HOME=/mnt/workspace/yangyandan/cache/huggingface,HF_ENDPOINT=https://hf-mirror.com"


# args="--config_yaml ./examples/LIBERO/train_files/starvla_cotrain_libero.yaml \
#       --framework.name QwenGR00TSpatial \
#       --framework.use_mv_images False \
#       --framework.qwenvl.base_vlm /mnt/workspace/zengshuang.zs/checkpoints/Qwen3-VL-4B-Instruct \
#       --datasets.vla_data.data_root_dir /mnt/nas-data-3/yangyandan/libero \
#       --datasets.vla_data.data_mix libero_all \
#       --datasets.vla_data.per_device_batch_size 16 \
#       --trainer.vla_data.video_backend torchvision_av \
#       --trainer.freeze_modules 'spatial_model,qwen_image_edit_model' \
#       --trainer.max_train_steps 30000 \
#       --trainer.save_interval 1000 \
#       --trainer.logging_frequency 100 \
#       --trainer.eval_interval 1000 \
#       --run_root_dir /mnt/workspace/junjin/code/starVLA/checkpoints \
#       --run_id 1227_liberoall_Qwen3vlGR00T_vggt_qwen_image_edit_2509_cross \
#       --wandb_entity junjin \
#       --wandb_project 1227_liberoall_Qwen3vlGR00T_vggt_qwen_image_edit_2509_cross\
#       --trainer.is_resume false \
#       --framework.fuser.type cross_attention \
#       "
#       # --trainer.resume_from_checkpoint null \

# # 打印将要传递的参数，方便调试
# echo "即将传递给训练脚本的参数："
# echo "${args}"
# echo ""
# echo "即将设置的环境变量："
# echo "${ENVS}"
# echo ""

# #amap_app_common_h20_nm125
# nebulactl run mdl --queue=amap_app_common_h20_na175 \
#                   --entry="starVLA/training/train_starvla.py" \
#                   --algo_name=pytorch260 \
#                   --worker_count=32 \
#                   --user_params="$args" \
#                   --file.cluster_file=./cluster.json \
#                   --job_name="1227_liberoall_Qwen3vlGR00T_vggt_qwen_image_edit_2509_cross" \
#                   --nas_file_system_id=1fff449945-wau24.cn-beijing.nas.aliyuncs.com,92bcb4b594-nvt70.cn-zhangjiakou.nas.aliyuncs.com,29016449f1c-mkq60.cn-wulanchabu.nas.aliyuncs.com,9dc4e499f2-tek11.cn-zhangjiakou.nas.aliyuncs.com,29e2cf482cb-cxw73.cn-wulanchabu.nas.aliyuncs.com \
#                   --nas_file_system_mount_path=/mnt/nas-data-5,/mnt/workspace,/mnt/nas-data-3,/mnt/nas-data-1,/mnt/xlab-nas-1 \
#                   --env="${ENVS}"


args="--config_yaml ./examples/LIBERO/train_files/starvla_cotrain_libero.yaml \
      --framework.name QwenGR00TDPT \
      --framework.use_mv_images False \
      --framework.qwenvl.base_vlm /mnt/workspace/zengshuang.zs/checkpoints/Qwen3-VL-4B-Instruct \
      --datasets.vla_data.data_root_dir /mnt/xlab-nas-1/junjin/dataset/libero_no_noops_1.0.0_lerobot \
      --datasets.vla_data.data_mix libero_depth_spatial \
      --datasets.vla_data.per_device_batch_size 16 \
      --trainer.vla_data.video_backend torchvision_av \
      --trainer.freeze_modules 'qwen_vl_interface.model,spatial_model,fuser,spatial_projector,action_model' \
      --trainer.max_train_steps 10000 \
      --trainer.save_interval 1000 \
      --trainer.logging_frequency 100 \
      --trainer.eval_interval 1000 \
      --run_root_dir /mnt/workspace/junjin/code/starVLA/checkpoints \
      --run_id 1230_libero_spatial_train_depth_Qwen3vlGR00T_vggt_cross \
      --wandb_entity junjin \
      --wandb_project 1230_libero_spatial_train_depth_Qwen3vlGR00T_vggt_cross\
      --trainer.is_resume false \
      --framework.fuser.type cross_attention \
      --framework.qwen_image_edit_model null \
      --trainer.pretrained_checkpoint /mnt/workspace/junjin/code/starVLA/checkpoints/1219_liberoall_Qwen3vlGR00T_vggt_cross/checkpoints/steps_30000_pytorch_model.pt\
      --trainer.reload_modules 'qwen_vl_interface.model,spatial_model,fuser,spatial_projector,action_model'
      "
      # --trainer.resume_from_checkpoint null \

# 打印将要传递的参数，方便调试
echo "即将传递给训练脚本的参数："
echo "${args}"
echo ""
echo "即将设置的环境变量："
echo "${ENVS}"
echo ""

#amap_app_common_h20_nm125
nebulactl run mdl --queue=amap_app_common_h20_na175 \
                  --entry="starVLA/training/train_starvla_dpt.py" \
                  --algo_name=pytorch260 \
                  --worker_count=16 \
                  --user_params="$args" \
                  --file.cluster_file=./cluster.json \
                  --job_name="1230_libero_spatial_train_depth_Qwen3vlGR00T_vggt_cross" \
                  --nas_file_system_id=1fff449945-wau24.cn-beijing.nas.aliyuncs.com,92bcb4b594-nvt70.cn-zhangjiakou.nas.aliyuncs.com,29016449f1c-mkq60.cn-wulanchabu.nas.aliyuncs.com,9dc4e499f2-tek11.cn-zhangjiakou.nas.aliyuncs.com,29e2cf482cb-cxw73.cn-wulanchabu.nas.aliyuncs.com \
                  --nas_file_system_mount_path=/mnt/nas-data-5,/mnt/workspace,/mnt/nas-data-3,/mnt/nas-data-1,/mnt/xlab-nas-1 \
                  --env="${ENVS}"
