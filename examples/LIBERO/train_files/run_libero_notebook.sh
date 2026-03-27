eval "$(conda shell.bash hook)"
# source activate
conda activate python3.10

run_id=0327_liberoall_Qwen3vlGR00T_vggt_longcat_view2_cross_cross_bs16_4gpus

# === Please modify the following paths according to your environment ===
Framework_name=QwenGR00TSpatial #QwenGR00T, QwenGR00TSpatial
batch_size=16
# freeze_module_list="qwen_vl_interface.model,spatial_model,qwen_image_edit_model.text_encoder,qwen_image_edit_model.transformer,qwen_image_edit_model.vae"
freeze_module_list="spatial_model,image_edit_model"
base_vlm=/mnt/workspace/zengshuang.zs/checkpoints/Qwen3-VL-4B-Instruct-Action
config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero.yaml
libero_data_root=/mnt/nas-data-3/yangyandan/libero
data_mix=libero_all # libero_all
run_root_dir=/mnt/workspace/junjin/code/starVLA/checkpoints
# === End of environment variable configuration ===
###########################################################################################



# export CUDA_HOME=/usr/local/cuda-12
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# export PATH=$CUDA_HOME/bin:$PATH
export WANDB_MODE=offline
export HF_HOME=/mnt/workspace/yangyandan/cache/huggingface 
export HF_ENDPOINT=https://hf-mirror.com 
export CHECKPOINT_BASEDIR=/mnt/workspace/zengshuang.zs/checkpoints

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
# cp $0 ${output_dir}/


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



accelerate launch starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${libero_data_root}\
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size ${batch_size} \
  --datasets.vla_data.num_workers 4 \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 30000 \
  --trainer.save_interval 5000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 1000 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_entity junjin \
  --wandb_project ${run_id}\
  --framework.fuser.type cross_attention \
  --framework.image_edit_model.view_num 2 \
  --framework.image_edit_model.fuser_type cross_attention 
  # --trainer.pretrained_checkpoint /mnt/workspace/zengshuang.zs/output/pretrain/1223_oxe_pretrain_Qwen3VL4BFast/checkpoints/steps_48000_pytorch_model.pt \
  # --trainer.reload_modules qwen_vl_interface \
  # --framework.image_edit_model null
  # --trainer.pretrained_checkpoint /mnt/workspace/junjin/code/starVLA/checkpoints/0109_liberoall_Qwen3vlGR00T_no_vggt_longcat_image_edit_cross_bs4_test2/checkpoints/steps_1000_pytorch_model.pt
    # --framework.spatial_model null \
  # --is_debug True


