#!/bin/bash
eval "$(conda shell.bash hook)"
# source activate
conda activate python3.10

task_suite_name=$1
start_idx=$2
end_idx=$3

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
    python generate_mv_videos.py \
    --dataset_name $task_suite_name \
    --start $current_start \
    --end $current_end &
    # 更新下一次的起始位置
    current_start=$current_end

    # 如果已经到达 end_idx，提前退出（防止空区间）
    if [ $current_start -ge $end_idx ]; then
        break
    fi
done


wait
