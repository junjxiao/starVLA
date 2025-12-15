

# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_IB_HCA=mlx5_2,mlx5_3

# used for check save when communication
# export NCCL_BLOCKING_WAIT=1
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_TIMEOUT=10000  # timeout set to 1 hour (unit: seconds)
# export NCCL_SOCKET_TIMEOUT_MS=360000
###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=QwenGR00T
freeze_module_list="qwen_vl_interface.model.model"
base_vlm=/mnt/workspace/zengshuang.zs/checkpoints/Qwen3-VL-4B-Instruct
config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero.yaml
libero_data_root=/mnt/nas-data-3/yangyandan/libero
data_mix=libero_10
run_root_dir=/mnt/workspace/zengshuang.zs/output/test
run_id=1212_QwenGR00T
# === End of environment variable configuration ===
###########################################################################################


export WANDB_MODE=disabled
export WANDB_MODE=offline
export WANDB_DISABLED=true
export CUDA_HOME=/usr/local/cuda-12
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=/mnt/workspace/yangyandan/cache/huggingface 
export HF_ENDPOINT=https://hf-mirror.com 
export CHECKPOINT_BASEDIR=/mnt/workspace/zengshuang.zs/checkpoints

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/


# CUDA_VISIBLE_DEVICES=3 accelerate launch \
#   --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
#   --num_processes 1 \
#   starVLA/training/train_starvla.py \
#   --config_yaml ${config_yaml} \
#   --framework.name ${Framework_name} \
#   --framework.qwenvl.base_vlm ${base_vlm} \
#   --datasets.vla_data.data_root_dir ${libero_data_root}\
#   --datasets.vla_data.data_mix ${data_mix} \
#   --datasets.vla_data.per_device_batch_size 2 \
#   --trainer.vla_data.video_backend torchvision_av \
#   --trainer.freeze_modules ${freeze_module_list} \
#   --trainer.max_train_steps 100000 \
#   --trainer.save_interval 10000 \
#   --trainer.logging_frequency 100 \
#   --trainer.eval_interval 1000 \
#   --run_root_dir ${run_root_dir} \
#   --run_id ${run_id} \
#   # --is_debug True



CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 \
  starVLA/training/train_starvla.py \
  --deepspeed starVLA/config/deepseeds/zero3.json \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${libero_data_root}\
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 2 \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 1000 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  # --is_debug True


