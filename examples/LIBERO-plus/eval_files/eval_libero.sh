#!/bin/bash

###########################################################################################
# === Please modify the following paths according to your environment ===
export LIBERO_HOME=/mnt/workspace/junjin/code/LIBERO-plus
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export LIBERO_Python=/mnt/workspace/junjin/conda/libero_plus/bin/python
export MUJOCO_GL=osmesa
export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME} # let eval_libero find the LIBERO tools
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo


host="127.0.0.1"
base_port=9879
unnorm_key="franka"
your_ckpt=/mnt/workspace/junjin/code/starVLA/checkpoints/1222_liberoall_Qwen3vlGR00T_vggt_mlayer/checkpoints/steps_20000_pytorch_model.pt
output_dir=/mnt/workspace/junjin/code/starVLA/outputs/libero-plus/libero4in1_Qwen3vlGR00T_vggt_mlayer_step30000
folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
# === End of environment variable configuration ===
###########################################################################################

# export DEBUG=true

LOG_DIR="${output_dir}/logs/$(date +"%Y%m%d_%H%M%S")"
mkdir -p ${LOG_DIR}


task_suite_name=libero_goal
num_trials_per_task=1
video_out_path="${output_dir}/${task_suite_name}/${folder_name}"
log_file="${LOG_DIR}/${task_suite_name}.log"

CUDA_VISIBLE_DEVICES=3 ${LIBERO_Python} ./examples/LIBERO-plus/eval_files/eval_libero.py \
    --args.pretrained-path ${your_ckpt} \
    --args.host "$host" \
    --args.port $base_port \
    --args.task-suite-name "$task_suite_name" \
    --args.num-trials-per-task "$num_trials_per_task" \
    --args.video-out-path "$video_out_path" \
    --args.log-path "$LOG_DIR" \
    2>&1 | tee "${log_file}" &


##########  eval libero_spatial ##########

# set it in background to run multiple evals in parallel with &

task_suite_name=libero_spatial
num_trials_per_task=1
video_out_path="${output_dir}/${task_suite_name}/${folder_name}"
log_file="${LOG_DIR}/${task_suite_name}.log"

CUDA_VISIBLE_DEVICES=3 ${LIBERO_Python} ./examples/LIBERO-plus/eval_files/eval_libero.py \
    --args.pretrained-path ${your_ckpt} \
    --args.host "$host" \
    --args.port $base_port \
    --args.task-suite-name "$task_suite_name" \
    --args.num-trials-per-task "$num_trials_per_task" \
    --args.video-out-path "$video_out_path" \
    2>&1 | tee "${log_file}" &


##########  eval libero_object ##########

task_suite_name=libero_object
num_trials_per_task=1
video_out_path="${output_dir}/${task_suite_name}/${folder_name}"
log_file="${LOG_DIR}/${task_suite_name}.log"

CUDA_VISIBLE_DEVICES=3 ${LIBERO_Python} ./examples/LIBERO-plus/eval_files/eval_libero.py \
    --args.pretrained-path ${your_ckpt} \
    --args.host "$host" \
    --args.port $base_port \
    --args.task-suite-name "$task_suite_name" \
    --args.num-trials-per-task "$num_trials_per_task" \
    --args.video-out-path "$video_out_path" \
    2>&1 | tee "${log_file}" &



##########  eval libero_long ##########

task_suite_name=libero_10
num_trials_per_task=1
video_out_path="${output_dir}/${task_suite_name}/${folder_name}"
log_file="${LOG_DIR}/${task_suite_name}.log"

CUDA_VISIBLE_DEVICES=3 ${LIBERO_Python} ./examples/LIBERO-plus/eval_files/eval_libero.py \
    --args.pretrained-path ${your_ckpt} \
    --args.host "$host" \
    --args.port $base_port \
    --args.task-suite-name "$task_suite_name" \
    --args.num-trials-per-task "$num_trials_per_task" \
    --args.video-out-path "$video_out_path" \
    2>&1 | tee "${log_file}" &

# =============== 等待所有后台任务完成 ===============
echo "Waiting for all evaluation tasks to finish..."
wait  # 👈 关键！等待所有 & 启动的子进程结束

# =============== 聚合结果 ===============
echo "All tasks completed. Aggregating results..."
export LOG_DIR="${LOG_DIR}"
python ./examples/LIBERO-plus/eval_files/aggregate_results.py
