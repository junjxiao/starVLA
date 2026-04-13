#!/bin/bash
# DEEPSPEED_CONFIG_FILE=starVLA/config/deepseeds/zero0.json,
ENVS="CHECKPOINT_BASEDIR=/mnt/workspace/zengshuang.zs/checkpoints,WANDB_MODE=offline,HF_HOME=/mnt/workspace/yangyandan/cache/huggingface,HF_ENDPOINT=https://hf-mirror.com"



run_id=0413_robotwin_Qwen3vlGR00TAML_vggt_cross_mlp_gated_tranformer_bs4
args="--config_yaml ./examples/Robotwin/train_files/starvla_cotrain_robotwin_abs.yaml \
      --framework.name QwenGR00TSpatialAML \
      --framework.qwenvl.base_vlm /mnt/workspace/zengshuang.zs/checkpoints/Qwen3-VL-4B-Instruct-Action \
      --datasets.vla_data.data_root_dir /mnt/workspace/vla_dataset/benchmark \
      --datasets.vla_data.data_mix robotwin_mix \
      --datasets.vla_data.per_device_batch_size 4 \
      --trainer.vla_data.video_backend torchvision_av \
      --trainer.freeze_modules 'spatial_model,image_edit_model' \
      --trainer.max_train_steps 150000 \
      --trainer.save_interval 1000 \
      --trainer.logging_frequency 100 \
      --trainer.eval_interval 1000 \
      --run_root_dir /mnt/workspace/junjin/code/starVLA/checkpoints \
      --run_id ${run_id} \
      --wandb_entity junjin \
      --wandb_project ${run_id}\
      --framework.fuser.type cross_attention \
      --framework.image_edit_model.view_num 2 \
      --framework.image_edit_model.fuser_type mlp_gated_tranformer \
      --trainer.pretrained_checkpoint /mnt/workspace/lintong.lt/output/vla_pretrain/0323_pretrain_Qwen3VL4BFast_bs2048/checkpoints/steps_8000_pytorch_model.pt \
      --trainer.reload_modules qwen_vl_interface \
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
nebulactl run mdl --queue=amap-poi_ppu810e \
                  --entry="starVLA/training/train_starvla.py" \
                  --algo_name=pytorch280 \
                  --worker_count=48 \
                  --user_params="$args" \
                  --file.cluster_file=./cluster.json \
                  --job_name="${run_id}" \
                  --nas_file_system_id=92bcb4b594-nvt70.cn-zhangjiakou.nas.aliyuncs.com,29016449f1c-mkq60.cn-wulanchabu.nas.aliyuncs.com,9dc4e499f2-tek11.cn-zhangjiakou.nas.aliyuncs.com,29e2cf482cb-cxw73.cn-wulanchabu.nas.aliyuncs.com \
                  --nas_file_system_mount_path=/mnt/workspace,/mnt/nas-data-3,/mnt/nas-data-1,/mnt/xlab-nas-1 \
                  --env="${ENVS}"
