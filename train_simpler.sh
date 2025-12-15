#!/bin/bash

ENVS="CHECKPOINT_BASEDIR=/mnt/workspace/zengshuang.zs/checkpoints,WANDB_MODE=offline,WANDB_MODE=disabled,HF_HOME=/mnt/workspace/yangyandan/cache/huggingface,HF_ENDPOINT=https://hf-mirror.com"


args="--deepspeed starVLA/config/deepseeds/zero0.json \
      --config_yaml ./examples/SimplerEnv/train_files/starvla_cotrain_oxe.yaml \
      --framework.name QwenGR00T \
      --framework.qwenvl.base_vlm /mnt/workspace/zengshuang.zs/checkpoints/Qwen3-VL-4B-Instruct \
      --datasets.vla_data.data_root_dir /mnt/workspace/zengshuang.zs/data/oxe \
      --datasets.vla_data.data_mix bridge_rt_1 \
      --datasets.vla_data.per_device_batch_size 8 \
      --trainer.vla_data.video_backend torchvision_av \
      --trainer.freeze_modules '' \
      --trainer.max_train_steps 20000 \
      --trainer.save_interval 2000 \
      --trainer.logging_frequency 100 \
      --trainer.eval_interval 1000 \
      --run_root_dir /mnt/workspace/zengshuang.zs/output/bridge_rt_1_na175 \
      --run_id 1214_bridge_rt_1_QwenGR00T \
      "

# 打印将要传递的参数，方便调试
echo "即将传递给训练脚本的参数："
echo "${args}"
echo ""
echo "即将设置的环境变量："
echo "${ENVS}"
echo ""

#amap_app_common_h20_nm125
nebulactl run mdl --queue=amap_app_common_h20_na175 \
                  --entry="starVLA/training/train_starvla.py" \
                  --algo_name=pytorch260 \
                  --worker_count=64 \
                  --user_params="$args" \
                  --file.cluster_file=./cluster.json \
                  --job_name="starVLA" \
                  --nas_file_system_id=1fff449945-wau24.cn-beijing.nas.aliyuncs.com,92bcb4b594-nvt70.cn-zhangjiakou.nas.aliyuncs.com,29016449f1c-mkq60.cn-wulanchabu.nas.aliyuncs.com,9dc4e499f2-tek11.cn-zhangjiakou.nas.aliyuncs.com \
                  --nas_file_system_mount_path=/mnt/nas-data-5,/mnt/workspace,/mnt/nas-data-3,/mnt/nas-data-1 \
                  --env="${ENVS}"



# #!/bin/bash

# ENVS="CHECKPOINT_BASEDIR=/mnt/workspace/zengshuang.zs/checkpoints,WANDB_MODE=offline,WANDB_MODE=disabled,HF_HOME=/mnt/workspace/yangyandan/cache/huggingface,HF_ENDPOINT=https://hf-mirror.com"


# args="--deepspeed starVLA/config/deepseeds/zero0.json \
#       --config_yaml ./examples/LIBERO/train_files/starvla_cotrain_libero.yaml \
#       --framework.name QwenGR00T \
#       --framework.qwenvl.base_vlm /mnt/workspace/zengshuang.zs/checkpoints/Qwen3-VL-4B-Instruct \
#       --datasets.vla_data.data_root_dir /mnt/nas-data-3/yangyandan/libero \
#       --datasets.vla_data.data_mix libero_all \
#       --datasets.vla_data.per_device_batch_size 16 \
#       --trainer.vla_data.video_backend torchvision_av \
#       --trainer.freeze_modules '' \
#       --trainer.max_train_steps 300000 \
#       --trainer.save_interval 10000 \
#       --trainer.logging_frequency 100 \
#       --trainer.eval_interval 1000 \
#       --run_root_dir /mnt/workspace/zengshuang.zs/output/test125 \
#       --run_id 1212_libero4in1_QwenGR00T \
#       "

# # 打印将要传递的参数，方便调试
# echo "即将传递给训练脚本的参数："
# echo "${args}"
# echo ""
# echo "即将设置的环境变量："
# echo "${ENVS}"
# echo ""

# #amap_app_common_h20_na175
# nebulactl run mdl --queue=amap_app_common_h20_nm125 \
#                   --entry="starVLA/training/train_starvla.py" \
#                   --algo_name=pytorch260 \
#                   --worker_count=2 \
#                   --user_params="$args" \
#                   --file.cluster_file=./cluster.json \
#                   --job_name="starVLA" \
#                   --nas_file_system_id=1fff449945-wau24.cn-beijing.nas.aliyuncs.com,92bcb4b594-nvt70.cn-zhangjiakou.nas.aliyuncs.com,29016449f1c-mkq60.cn-wulanchabu.nas.aliyuncs.com,9dc4e499f2-tek11.cn-zhangjiakou.nas.aliyuncs.com \
#                   --nas_file_system_mount_path=/mnt/nas-data-5,/mnt/workspace,/mnt/nas-data-3,/mnt/nas-data-1 \
#                   --env="${ENVS}"
