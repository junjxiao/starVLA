#!/bin/bash
eval "$(conda shell.bash hook)"
# source activate
conda activate /mnt/workspace/junjin/conda/starvla
###########################################################################################
# === Please modify the following paths according to your environment ===
export LIBERO_HOME=/mnt/workspace/junjin/code/LIBERO-plus
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export LIBERO_Python=/mnt/workspace/junjin/conda/starvla/bin/python
export MUJOCO_GL=osmesa
export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME} # let eval_libero find the LIBERO tools
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo


unnorm_key="franka"
your_ckpt=/mnt/workspace/zengshuang.zs/output/libero_all/0104_libero4in1_QwenGR00T_sft/checkpoints/steps_30000_pytorch_model.pt
output_dir=/mnt/workspace/junjin/code/starVLA/outputs/libero-plus/0104_libero4in1_QwenGR00T_sft_step30000
# === End of environment variable configuration ===
###########################################################################################


base_port=9882
task_suite_name=$1
echo $task_suite_name
num_trials_per_task=1

# torchrun --nproc_per_node=$gpu_per_pod --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT ./examples/LIBERO-plus/eval_files/eval_nebula/eval_libero_model.py \
torchrun --nproc_per_node=1 ./examples/LIBERO-plus/eval_files/eval_nebula/eval_libero_model.py \
    --pretrained_path $your_ckpt \
    --task_suite_name $task_suite_name \
    --num_trials_per_task $num_trials_per_task \
    --output_dir $output_dir \
    --start_idx 0 \
    --end_idx 5


# # =============== 聚合结果 ===============
# echo "All tasks completed. Aggregating results..."
# export LOG_DIR="${LOG_DIR}"
# python ./examples/LIBERO-plus/eval_files/aggregate_results.py