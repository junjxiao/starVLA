# #!/bin/bash
# eval "$(conda shell.bash hook)"
# # source activate
# conda activate python3.10

# task_suite_name=$1
# start_idx=$2
# end_idx=$3

# total=$((end_idx - start_idx))
# chunk_size=$((total / 2))
# remainder=$((total % 2))
# current_start=$start_idx

# for i in {0..1}; do
#     # 前 'remainder' 份多分配1个元素（用于处理不能整除的情况）
#     if [ $i -lt $remainder ]; then
#         current_end=$((current_start + chunk_size + 1))
#     else
#         current_end=$((current_start + chunk_size))
#     fi

#     # 确保最后一份不超过 end_idx（安全边界）
#     if [ $current_end -gt $end_idx ]; then
#         current_end=$end_idx
#     fi

#     echo "Part $((i)): start=$current_start, end=$current_end ([$current_start, $current_end))"
#     # torchrun --nproc_per_node=$gpu_per_pod --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$((MASTER_ADDR+i)) --master_port=$MASTER_PORT 
#     python generate_mv_videos.py \
#     --dataset_name $task_suite_name \
#     --start $current_start \
#     --end $current_end &
#     # 更新下一次的起始位置
#     current_start=$current_end

#     # 如果已经到达 end_idx，提前退出（防止空区间）
#     if [ $current_start -ge $end_idx ]; then
#         break
#     fi
# done


# wait
#!/bin/bash
# eval "$(conda shell.bash hook)"
# conda activate python3.10

dataset_name=$1
chunk_name=$2
global_start=${3:-0}
global_end=${4:-1} 
# 新增参数：并行进程数，默认为 1 (串行，最稳)，单卡建议设为 1 或 2
NUM_PROCESSES=${5:-2} 

echo "========================================="
echo "Starting Single-GPU Multi-Process Processing"
echo "Dataset: $dataset_name"
echo "Chunk:   $chunk_name"
echo "Procs:   $NUM_PROCESSES"
echo "========================================="

# 1.确定原始视频目录
ORIG_CHUNK_DIR="/mnt/xlab-nas-2/vla_dataset/lerobot/oxe/${dataset_name}/videos/${chunk_name}/observation.images.image_0"

if [ ! -d "$ORIG_CHUNK_DIR" ]; then
    echo "Error: Directory $ORIG_CHUNK_DIR does not exist."
    exit 1
fi

# 2. 获取视频文件列表并计数
mapfile -t ALL_VIDEOS < <(ls "$ORIG_CHUNK_DIR" | grep -E '\.(mp4|avi|mov|mkv)$' | sort)
TOTAL_VIDEOS=${#ALL_VIDEOS[@]}

if [ "$TOTAL_VIDEOS" -eq 0 ]; then
    echo "Warning: No videos found in $ORIG_CHUNK_DIR"
    exit 0
fi

# 处理全局范围参数
if [ "$global_end" -eq -1 ] || [ "$global_end" -gt "$TOTAL_VIDEOS" ]; then
    global_end=$TOTAL_VIDEOS
fi

START_IDX=$global_start
END_IDX=$global_end
TOTAL_TO_PROCESS=$((END_IDX - START_IDX))

echo "Total videos in chunk: $TOTAL_VIDEOS"
echo "Processing range: [$START_IDX, $END_IDX) -> Count: $TOTAL_TO_PROCESS"

if [ "$TOTAL_TO_PROCESS" -le 0 ]; then
    echo "No videos to process in the specified range."
    exit 0
fi

# 3. 计算每个进程的任务分配
BASE_COUNT=$((TOTAL_TO_PROCESS / NUM_PROCESSES))
REMAINDER=$((TOTAL_TO_PROCESS % NUM_PROCESSES))

CURRENT_START=$START_IDX
PIDS=()

echo "Launching $NUM_PROCESSES processes on GPU 0..."

for (( i=0; i<NUM_PROCESSES; i++ )); do
    # 计算当前进程的处理数量
    if [ $i -lt $REMAINDER ]; then
        COUNT=$((BASE_COUNT + 1))
    else
        COUNT=$BASE_COUNT
    fi
    
    CURRENT_END=$((CURRENT_START + COUNT))
    
    # 确保不超过全局结束位置
    if [ $CURRENT_END -gt $END_IDX ]; then
        CURRENT_END=$END_IDX
    fi

    # 如果数量为0，跳过
    if [ $CURRENT_START -ge $CURRENT_END ]; then
        continue
    fi

    echo "Launching Process $i: Videos [$CURRENT_START, $CURRENT_END)"

    # 启动后台进程
    # 注意：所有进程都使用 cuda:0
    CUDA_VISIBLE_DEVICES=0 python generate_mv_videos.py \
        --dataset_name "$dataset_name" \
        --chunk_name "$chunk_name" \
        --start "$CURRENT_START" \
        --end "$CURRENT_END" \
        &
    
    PIDS+=($!)
    
    CURRENT_START=$CURRENT_END
done

echo "All processes launched. Waiting for completion..."

# 4. 等待所有后台进程结束
FAIL=0
for pid in "${PIDS[@]}"; do
    wait $pid
    if [ $? -ne 0 ]; then
        echo "Process $pid failed!"
        FAIL=1
    fi
done

if [ $FAIL -eq 0 ]; then
    echo "✅ All processes finished successfully for chunk $chunk_name."
else
    echo "❌ One or more processes failed."
    exit 1
fi
