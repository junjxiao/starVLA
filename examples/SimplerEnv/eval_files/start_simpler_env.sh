#!/bin/bash

echo `which python`

export SimplerEnv_PATH=/mnt/workspace/zengshuang.zs/SimplerEnv
export PYTHONPATH=/mnt/workspace/junjin/conda/simpler_env:${PYTHONPATH}
export PYTHONPATH=$(pwd):${PYTHONPATH}

MODEL_PATH=/mnt/workspace/junjin/code/starVLA/checkpoints/1216_simpler_Qwen3vlGR00T_vggt_concat/checkpoints/steps_28000_pytorch_model.pt #/mnt/workspace/zengshuang.zs/checkpoints/Qwen-GR00T-Bridge/checkpoints/steps_45000_pytorch_model.pt
logging_dir=/mnt/workspace/junjin/code/starVLA/outputs/simpler/simpler_Qwen3vlGR00T_vggt_concat_step28000
ckpt_path=${MODEL_PATH}
TSET_NUM=3
# export DEBUG=1

port=56706

# IFS=',' read -r -a CUDA_DEVICES <<< "$CUDA_VISIBLE_DEVICES"
# NUM_GPUS=${#CUDA_DEVICES[@]} 

# echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
# echo "CUDA_DEVICES: ${CUDA_DEVICES[@]}"
# echo "NUM_GPUS: $NUM_GPUS"

scene_name=bridge_table_1_v1
robot=widowx
rgb_overlay_path=${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png
robot_init_x=0.147
robot_init_y=0.028

declare -a ENV_NAMES=(
  StackGreenCubeOnYellowCubeBakedTexInScene-v0
  PutCarrotOnPlateInScene-v0
  PutSpoonOnTableClothInScene-v0
)



for i in "${!ENV_NAMES[@]}"; do
  env="${ENV_NAMES[i]}"
  for ((run_idx=1; run_idx<=TSET_NUM; run_idx++)); do
    log_file="${logging_dir}/${env}_run${run_idx}_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p "${logging_dir}"
    echo "▶️ Launching task [${env}] run#${run_idx} on GPU $gpu_id, log → ${log_file}"

    CUDA_VISIBLE_DEVICES=3 python examples/SimplerEnv/eval_files/start_simpler_env.py \
      --ckpt-path ${ckpt_path} \
      --port ${port} \
      --robot ${robot} \
      --policy-setup widowx_bridge \
      --control-freq 5 \
      --sim-freq 500 \
      --max-episode-steps 120 \
      --env-name "${env}" \
      --scene-name ${scene_name} \
      --rgb-overlay-path ${rgb_overlay_path} \
      --robot-init-x ${robot_init_x} ${robot_init_x} 1 \
      --robot-init-y ${robot_init_y} ${robot_init_y} 1 \
      --obj-variation-mode episode \
      --obj-episode-range 0 24 \
      --logging-dir ${logging_dir} \
      --robot-init-rot-quat-center 0 0 0 1 \
      --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
      2>&1 | tee "${log_file}"

  done
done

declare -a ENV_NAMES_V2=(
  PutEggplantInBasketScene-v0
)

scene_name=bridge_table_1_v2
robot=widowx_sink_camera_setup
rgb_overlay_path=${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png
robot_init_x=0.127
robot_init_y=0.06

for i in "${!ENV_NAMES_V2[@]}"; do
  env="${ENV_NAMES_V2[i]}"
  for ((run_idx=1; run_idx<=TSET_NUM; run_idx++)); do
    log_file="${logging_dir}/${env}_run${run_idx}_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p "${logging_dir}"
    echo "▶️ Launching V2 task [${env}] run#${run_idx} on GPU $gpu_id, log → ${log_file}"

    CUDA_VISIBLE_DEVICES=3 python examples/SimplerEnv/eval_files/start_simpler_env.py \
      --ckpt-path ${ckpt_path} \
      --port ${port} \
      --robot ${robot} \
      --policy-setup widowx_bridge \
      --control-freq 5 \
      --sim-freq 500 \
      --max-episode-steps 120 \
      --env-name "${env}" \
      --scene-name ${scene_name} \
      --rgb-overlay-path ${rgb_overlay_path} \
      --robot-init-x ${robot_init_x} ${robot_init_x} 1 \
      --robot-init-y ${robot_init_y} ${robot_init_y} 1 \
      --obj-variation-mode episode \
      --obj-episode-range 0 24 \
      --logging-dir ${logging_dir} \
      --robot-init-rot-quat-center 0 0 0 1 \
      --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
      2>&1 | tee "${log_file}"
  done
done

echo "✅ Finished"
