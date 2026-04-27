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
your_ckpt=/mnt/workspace/junjin/code/starVLA/checkpoints/0423_liberoall_Qwen3vlGR00TAML_vggt_longcat_view2_cross_mlp_gated_tranformer_ck10_JAT2048_14k_bs16_4gpus_reload_vlm_action/checkpoints/steps_35000_pytorch_model.pt
output_dir=/mnt/workspace/junjin/code/starVLA/outputs/libero-plus/0423_liberoall_Qwen3vlGR00TAML_vggt_longcat_view2_cross_mlp_gated_tranformer_ck10_JAT2048_14k_bs16_4gpus_reload_vlm_action_step35000
# === End of environment variable configuration ===
###########################################################################################

task_suite_name=$1
start_idx=$2
end_idx=$3
num_trials_per_task=1
# torchrun --nproc_per_node=1 ./examples/LIBERO-plus/eval_files/eval_nebula/eval_libero_model.py \

total=$((end_idx - start_idx))
chunk_size=$((total / 2))
remainder=$((total % 2))
current_start=$start_idx

for i in {0..1}; do
    # 前 'remainder' 份多分配1个元素（用于处理不能整除的情况）
    if [ $i -lt $remainder ]; then
        current_end=$((current_start + chunk_size + 1))
    else
        current_end=$((current_start + chunk_size))
    fi

    # 确保最后一份不超过 end_idx（安全边界）
    if [ $current_end -gt $end_idx ]; then
        current_end=$end_idx
    fi

    echo "Part $((i)): start=$current_start, end=$current_end ([$current_start, $current_end))"
    # torchrun --nproc_per_node=$gpu_per_pod --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$((MASTER_ADDR+i)) --master_port=$MASTER_PORT 
    python ./examples/LIBERO-plus/eval_files/eval_nebula/eval_libero_model.py \
    --pretrained_path $your_ckpt \
    --task_suite_name $task_suite_name \
    --num_trials_per_task $num_trials_per_task \
    --output_dir $output_dir \
    --start_idx $current_start \
    --end_idx $current_end &
    # 更新下一次的起始位置
    current_start=$current_end

    # 如果已经到达 end_idx，提前退出（防止空区间）
    if [ $current_start -ge $end_idx ]; then
        break
    fi
done


wait
# # =============== 聚合结果 ===============
# echo "All tasks completed. Aggregating results..."
# export LOG_DIR="${LOG_DIR}"
# python ./examples/LIBERO-plus/eval_files/aggregate_results.py