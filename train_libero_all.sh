#!/bin/bash

ENVS="CHECKPOINT_BASEDIR=/mnt/workspace/zengshuang.zs/checkpoints,WANDB_MODE=offline,HF_HOME=/mnt/workspace/yangyandan/cache/huggingface,HF_ENDPOINT=https://hf-mirror.com"

run_id=0408_liberoall_Qwen3vlGR00T_vggt_longcat_view2_cross_self_bs16_4gpus_nopretrain
args="--config_yaml ./examples/LIBERO/train_files/starvla_cotrain_libero.yaml \
      --framework.name QwenGR00TSpatial \
      --framework.qwenvl.base_vlm /mnt/workspace/zengshuang.zs/checkpoints/Qwen3-VL-4B-Instruct-Action \
      --datasets.vla_data.data_root_dir /mnt/nas-data-3/yangyandan/libero \
      --datasets.vla_data.data_mix libero_all \
      --datasets.vla_data.per_device_batch_size 16 \
      --trainer.vla_data.video_backend torchvision_av \
      --trainer.freeze_modules 'spatial_model,image_edit_model' \
      --trainer.max_train_steps 30000 \
      --trainer.save_interval 5000 \
      --trainer.logging_frequency 100 \
      --trainer.eval_interval 1000000 \
      --run_root_dir /mnt/workspace/junjin/code/starVLA/checkpoints \
      --run_id ${run_id} \
      --wandb_entity junjin \
      --wandb_project ${run_id}\
      --framework.fuser.type cross_attention \
      --framework.image_edit_model.view_num 2 \
      --framework.image_edit_model.fuser_type self_attention \
      --datasets.vla_data.mv_data_root_dir /mnt/xlab-nas-1/junjin/dataset/libero_mv_feats \
      --trainer.learning_rate.qwen_vl_interface 3.0e-05
      "
      # --trainer.pretrained_checkpoint /mnt/workspace/lintong.lt/output/vla_pretrain/0202_pretrain_Qwen3VL4BFast_GOAR_task_balance/checkpoints/steps_10000_pytorch_model.pt \
      # --trainer.reload_modules qwen_vl_interface \
      # --datasets.vla_data.mv_data_root_dir /mnt/xlab-nas-1/junjin/dataset/libero_mv_images \
      # --framework.spatial_model null \
      # --trainer.pretrained_checkpoint /mnt/workspace/zengshuang.zs/output/pretrain/1223_oxe_pretrain_Qwen3VL4BFast/checkpoints/steps_48000_pytorch_model.pt \
      # --trainer.reload_modules qwen_vl_interface
      # --trainer.pretrained_checkpoint /mnt/workspace/junjin/code/starVLA/checkpoints/0116_liberoall_Qwen3vlGR00T_vggt_longcat_image_edit_cross_bs16/checkpoints/steps_10000_pytorch_model.pt \
      # --trainer.resume_from_checkpoint null \


#amap_app_common_h20_nm125
# amap-poi_ppu810e
# amap_app_common_h20_na175
# amap_app_vtspoi_h20
nebulactl run mdl --queue=amap-poi_ppu810e \
                  --entry="starVLA/training/train_starvla.py" \
                  --algo_name=pytorch280 \
                  --worker_count=4 \
                  --user_params="$args" \
                  --file.cluster_file=./cluster.json \
                  --job_name="${run_id}" \
                  --nas_file_system_id=92bcb4b594-nvt70.cn-zhangjiakou.nas.aliyuncs.com,29016449f1c-mkq60.cn-wulanchabu.nas.aliyuncs.com,9dc4e499f2-tek11.cn-zhangjiakou.nas.aliyuncs.com,29e2cf482cb-cxw73.cn-wulanchabu.nas.aliyuncs.com \
                  --nas_file_system_mount_path=/mnt/workspace,/mnt/nas-data-3,/mnt/nas-data-1,/mnt/xlab-nas-1 \
                  --env="${ENVS}"

# nebulactl run mdl --queue=amap-poi_ppu810e \
#                   --entry="bash examples/LIBERO/train_files/run_libero_notebook.sh" \
#                   --algo_name=pytorch260 \
#                   --worker_count=1 \
#                   --user_params="" \
#                   --file.cluster_file=./cluster_docker.json \
#                   --job_name="starVLA" \
#                   --nas_file_system_id=92bcb4b594-nvt70.cn-zhangjiakou.nas.aliyuncs.com,29016449f1c-mkq60.cn-wulanchabu.nas.aliyuncs.com,9dc4e499f2-tek11.cn-zhangjiakou.nas.aliyuncs.com,29e2cf482cb-cxw73.cn-wulanchabu.nas.aliyuncs.com \
#                   --nas_file_system_mount_path=/mnt/workspace,/mnt/nas-data-3,/mnt/nas-data-1,/mnt/xlab-nas-1 \
#                   --custom_docker_image=hub.docker.alibaba-inc.com/mdl/notebook_saved:xiaojunjin.xjj_libero_plus_ppu2_20260119180727

# train depth head
# base_vlm='/mnt/workspace/zengshuang.zs/checkpoints/Qwen3-VL-4B-Instruct'
# freeze_modules='qwen_vl_interface.model,spatial_model,fuser,spatial_projector,action_model'
# run_id=1230_libero_spatial_train_depth_Qwen3vlGR00T_vggt_cross
# pretrained_checkpoint=/mnt/workspace/junjin/code/starVLA/checkpoints/1219_liberoall_Qwen3vlGR00T_vggt_cross/checkpoints/steps_30000_pytorch_model.pt
# # base_vlm='/mnt/workspace/zengshuang.zs/checkpoints/Qwen2.5-VL-3B-Instruct'
# # freeze_modules="qwen_vl_interface.model,action_model"
# # run_id=1230_libero_spatial_train_depth_Qwen3vlGR00T_orig
# # pretrained_checkpoint=/mnt/workspace/zengshuang.zs/checkpoints/Qwen2.5-VL-GR00T-LIBERO-4in1/checkpoints/steps_30000_pytorch_model.pt
# args="--config_yaml ./examples/LIBERO/train_files/starvla_cotrain_libero.yaml \
#       --framework.name QwenGR00TDPT \
#       --framework.use_mv_images False \
#       --framework.qwenvl.base_vlm ${base_vlm} \
#       --datasets.vla_data.data_root_dir /mnt/xlab-nas-1/junjin/dataset/libero_no_noops_1.0.0_lerobot \
#       --datasets.vla_data.data_mix libero_depth_spatial \
#       --datasets.vla_data.per_device_batch_size 16 \
#       --trainer.vla_data.video_backend torchvision_av \
#       --trainer.freeze_modules ${freeze_modules} \
#       --trainer.max_train_steps 10000 \
#       --trainer.save_interval 1000 \
#       --trainer.logging_frequency 100 \
#       --trainer.eval_interval 100 \
#       --run_root_dir /mnt/workspace/junjin/code/starVLA/checkpoints \
#       --run_id ${run_id} \
#       --wandb_entity junjin \
#       --wandb_project ${run_id}\
#       --trainer.is_resume false \
#       --framework.fuser.type cross_attention \
#       --framework.qwen_image_edit_model null \
#       --trainer.pretrained_checkpoint ${pretrained_checkpoint}\
#       --trainer.reload_modules ${freeze_modules} \
#       "
#       # --framework.spatial_model null
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
#                   --entry="starVLA/training/train_starvla_dpt.py" \
#                   --algo_name=pytorch260 \
#                   --worker_count=16 \
#                   --user_params="$args" \
#                   --file.cluster_file=./cluster.json \
#                   --job_name="${run_id}" \
#                   --nas_file_system_id=1fff449945-wau24.cn-beijing.nas.aliyuncs.com,92bcb4b594-nvt70.cn-zhangjiakou.nas.aliyuncs.com,29016449f1c-mkq60.cn-wulanchabu.nas.aliyuncs.com,9dc4e499f2-tek11.cn-zhangjiakou.nas.aliyuncs.com,29e2cf482cb-cxw73.cn-wulanchabu.nas.aliyuncs.com \
#                   --nas_file_system_mount_path=/mnt/nas-data-5,/mnt/workspace,/mnt/nas-data-3,/mnt/nas-data-1,/mnt/xlab-nas-1 \
#                   --env="${ENVS}"
