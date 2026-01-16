#!/bin/bash
eval "$(conda shell.bash hook)"
# source activate
conda activate python3.10
# pip install -r requirements.txt
# pip list
# ls /usr/lib64/libOSMesa.so*
###########################################################################################
# === Please modify the following paths according to your environment ===
export LIBERO_HOME=/mnt/workspace/junjin/code/LIBERO-plus
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export LIBERO_Python=/mnt/workspace/junjin/conda/starvla/bin/python
export MUJOCO_GL=osmesa
export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME} # let eval_libero find the LIBERO tools
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo


unnorm_key="franka"
your_ckpt=/mnt/workspace/junjin/code/starVLA/checkpoints/0114_liberoall_Qwen3vlGR00T_no_vggt_longcat_image_edit_cross_bs16/checkpoints/steps_20000_pytorch_model.pt
output_dir=/mnt/workspace/junjin/code/starVLA/outputs/libero-plus/0114_liberoall_Qwen3vlGR00T_no_vggt_longcat_image_edit_cross_bs16_step20000
# === End of environment variable configuration ===
###########################################################################################

task_suite_name=$1
start_idx=$2
end_idx=$3
num_trials_per_task=1
# torchrun --nproc_per_node=1 ./examples/LIBERO-plus/eval_files/eval_nebula/eval_libero_model.py \

torchrun --nproc_per_node=$gpu_per_pod --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT ./examples/LIBERO-plus/eval_files/eval_nebula/eval_libero_model.py \
    --pretrained_path $your_ckpt \
    --task_suite_name $task_suite_name \
    --num_trials_per_task $num_trials_per_task \
    --output_dir $output_dir \
    --start_idx $start_idx \
    --end_idx $end_idx


# # =============== 聚合结果 ===============
# echo "All tasks completed. Aggregating results..."
# export LOG_DIR="${LOG_DIR}"
# python ./examples/LIBERO-plus/eval_files/aggregate_results.py