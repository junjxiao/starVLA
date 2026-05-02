

# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_IB_HCA=mlx5_2,mlx5_3

# used for check save when communication
# export NCCL_BLOCKING_WAIT=1
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_TIMEOUT=1000  # timeout set to 1 hour (unit: seconds)

###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=QwenGR00TSpatialAML
freeze_module_list='qwen_vl_interface.model,spatial_model,image_edit_model'
base_vlm=/mnt/workspace/zengshuang.zs/checkpoints/Qwen3-VL-4B-Instruct-Action
config_yaml=./examples/Robotwin/train_files/starvla_cotrain_robotwin_abs.yaml
data_root=/mnt/workspace/vla_dataset/benchmark
run_root_dir=/mnt/workspace/junjin/code/starVLA/checkpoints
data_mix=robotwin_mix
run_id=test_robotwin
# === End of environment variable configuration ===
###########################################################################################


export WANDB_MODE=offline
export CUDA_HOME=/usr/local/cuda-12
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH
export HF_HOME=/mnt/workspace/yangyandan/cache/huggingface 
export HF_ENDPOINT=https://hf-mirror.com 
export CHECKPOINT_BASEDIR=/mnt/workspace/zengshuang.zs/checkpoints

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/

# --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
#   --num_processes 8 \
CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1\
  --master_port=29502\
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${data_root}\
  --datasets.vla_data.per_device_batch_size 1 \
  --datasets.vla_data.num_workers 0 \
  --datasets.vla_data.data_mix ${data_mix} \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 100 \
  --trainer.logging_frequency 1 \
  --trainer.eval_interval 1 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --framework.fuser.type 'cross_attention' \
  --framework.image_edit_model.view_num 2 \
  --framework.image_edit_model.fuser_type 'mlp_gated_tranformer' \
  # --trainer.pretrained_checkpoint /mnt/workspace/lintong.lt/output/vla_pretrain/0323_pretrain_Qwen3VL4BJAT_bs2048/checkpoints/steps_14000_pytorch_model.pt \
  # --trainer.reload_modules qwen_vl_interface,action_model \
  # --datasets.vla_data.mv_data_root_dir /mnt/xlab-nas-1/junjin/dataset/libero_mv_feats \



##### Multi-Server Multi-GPU training script #####
  # accelerate launch \
  #   --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  #   --main_process_ip $MASTER_ADDR \
  #   --main_process_port $MASTER_PORT \
  #   --machine_rank $SLURM_PROCID \
  #   --num_machines $SLURM_NNODES \
  #   --num_processes=${TOTAL_GPUS} \
  #   starVLA/training/train_starvla.py \
  #   --config_yaml ${config_yaml} \
  #   --framework.name ${Framework_name} \
  #   --framework.qwenvl.base_vlm ${base_vlm} \
  #   --run_root_dir ${run_root_dir} \
  #   --run_id ${run_id} \
  #   --wandb_project your_project \
  #   --wandb_entity your_name
##### Multi-Server Multi-GPU training script #####
