#!/bin/bash
eval "$(conda shell.bash hook)"
# source activate
conda activate python3.10
echo `which python`

export SimplerEnv_PATH=/mnt/workspace/zengshuang.zs/SimplerEnv
export PYTHONPATH=/opt/conda/envs/python3.10:${PYTHONPATH}
export PYTHONPATH=$(pwd):${PYTHONPATH}

MODEL_PATH=/mnt/workspace/junjin/code/starVLA/checkpoints/1222_simpler_Qwen3vlGR00T_vggt_cross/checkpoints/steps_20000_pytorch_model.pt #/mnt/workspace/zengshuang.zs/checkpoints/Qwen-GR00T-Bridge/checkpoints/steps_45000_pytorch_model.pt
logging_dir=/mnt/workspace/junjin/code/starVLA/outputs/simpler/1222_simpler_Qwen3vlGR00T_vggt_cross_step20000
ckpt_path=${MODEL_PATH}
TSET_NUM=1
# export DEBUG=1

port=56706

# 从命令行参数获取环境列表
ENV_NAMES=("$@")

# 全局配置（根据你的实际路径修改）

gpu_id=0
export CUDA_VISIBLE_DEVICES=${gpu_id}

# 创建日志目录
mkdir -p "${logging_dir}"

# 并行执行所有任务
for env in "${ENV_NAMES[@]}"; do
    for ((run_idx=1; run_idx<=TSET_NUM; run_idx++)); do
        # 根据环境名设置不同配置
        if [[ "$env" == "PutEggplantInBasketScene-v0" ]]; then
            # V2 配置
            scene_name="bridge_table_1_v2"
            robot="widowx_sink_camera_setup"
            rgb_overlay_path="${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png"
            robot_init_x=0.127
            robot_init_y=0.06
        else
            # V1 配置（默认）
            scene_name="bridge_table_1_v1"
            robot="widowx"
            rgb_overlay_path="${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png"
            robot_init_x=0.147
            robot_init_y=0.028
        fi

        log_file="${logging_dir}/${env}_run${run_idx}_$(date +%Y%m%d_%H%M%S).log"
        echo "▶️ Launching task [${env}] run#${run_idx} on GPU $gpu_id, log → ${log_file}"

        # 后台并行执行
        python examples/SimplerEnv/eval_files/eval_nebula/start_simpler_env.py \
          --ckpt-path "${ckpt_path}" \
          --port "${port}" \
          --robot "${robot}" \
          --policy-setup widowx_bridge \
          --control-freq 5 \
          --sim-freq 500 \
          --max-episode-steps 120 \
          --env-name "${env}" \
          --scene-name "${scene_name}" \
          --rgb-overlay-path "${rgb_overlay_path}" \
          --robot-init-x "${robot_init_x}" "${robot_init_x}" 1 \
          --robot-init-y "${robot_init_y}" "${robot_init_y}" 1 \
          --obj-variation-mode episode \
          --obj-episode-range 0 24 \
          --logging-dir "${logging_dir}" \
          --robot-init-rot-quat-center 0 0 0 1 \
          --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
          2>&1 | tee "${log_file}" &
    done
done

# 等待所有后台任务完成
wait
echo "✅ All tasks finished!"
