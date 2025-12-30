

# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_IB_HCA=mlx5_2,mlx5_3

# used for check save when communication
# export NCCL_BLOCKING_WAIT=1
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_TIMEOUT=10000  # timeout set to 1 hour (unit: seconds)
# export NCCL_SOCKET_TIMEOUT_MS=360000
###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=QwenGR00TDPT #QwenGR00T, QwenGR00TSpatial
# freeze_module_list="qwen_vl_interface.model,spatial_model,qwen_image_edit_model.text_encoder,qwen_image_edit_model.transformer,qwen_image_edit_model.vae"
freeze_module_list="qwen_vl_interface.model,spatial_model,fuser,spatial_projector,action_model"

base_vlm=/mnt/workspace/zengshuang.zs/checkpoints/Qwen3-VL-4B-Instruct
config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero.yaml
libero_data_root=/mnt/xlab-nas-1/junjin/dataset/libero_no_noops_1.0.0_lerobot
data_mix=libero_depth_spatial
run_root_dir=/mnt/workspace/junjin/code/starVLA/checkpoints
run_id='1230_libero_spatial_train_depth_Qwen3vlGR00T_vggt_cross'
# === End of environment variable configuration ===
###########################################################################################


export WANDB_MODE=offline
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



CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1\
  --master_port=29502\
  starVLA/training/train_starvla_dpt.py \
  --deepspeed starVLA/config/deepseeds/zero3.json \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${libero_data_root}\
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 1 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --framework.fuser.type 'cross_attention' \
  --framework.qwen_image_edit_model null \
  --trainer.pretrained_checkpoint /mnt/workspace/junjin/code/starVLA/checkpoints/1219_liberoall_Qwen3vlGR00T_vggt_cross/checkpoints/steps_30000_pytorch_model.pt\
  --trainer.reload_modules ${freeze_module_list}

  # --is_debug True


