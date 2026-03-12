#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate python3.10

# === 更改你的配置 ===
STARVLA_PATH=/mnt/workspace/junjin/code/starVLA
policy_ckpt_path=/mnt/workspace/zengshuang.zs/output/robotwin2/0130_robotwin2_QwenJAT_vggt_sft_64GPUs_chunk30/checkpoints/steps_35000_pytorch_model.pt
log_path=/mnt/workspace/junjin/code/starVLAPretrain/outputs/robotwin/0130_robotwin2_QwenJAT_vggt_sft_64GPUs_chunk30_step35000
task_config="demo_clean"
NUM_GPUS=8
TASKS_PER_GPU=3
# === === ===


ROBOTWIN_PATH=/mnt/workspace/junjin/code/RoboTwin
EVAL_FILES_PATH=$STARVLA_PATH/examples/Robotwin/eval_files/eval_nebula/
DEPLOY_POLICY_PATH=$EVAL_FILES_PATH/deploy_policy.yml
seed=0
policy_name="model2robotwin_interface"


MAX_CONCURRENT=$((NUM_GPUS * TASKS_PER_GPU))  # 24

export PYTHONPATH="$ROBOTWIN_PATH:$EVAL_FILES_PATH:$STARVLA_PATH:${PYTHONPATH}"
cd "$ROBOTWIN_PATH"

# 获取任务列表
task_names=(
    adjust_bottle
    beat_block_hammer
    blocks_ranking_rgb
    blocks_ranking_size
    click_alarmclock
    click_bell
    dump_bin_bigbin
    grab_roller
    handover_block
    handover_mic
    hanging_mug
    lift_pot
    move_can_pot
    move_pillbottle_pad
    move_playingcard_away
    move_stapler_pad
    open_laptop
    open_microwave
    pick_diverse_bottles
    pick_dual_bottles
    place_a2b_left
    place_a2b_right
    place_bread_basket
    place_bread_skillet
    place_burger_fries
    place_can_basket
    place_cans_plasticbox
    place_container_plate
    place_dual_shoes
    place_empty_cup
    place_fan
    place_mouse_pad
    place_object_basket
    place_object_scale
    place_object_stand
    place_phone_stand
    place_shoe
    press_stapler
    put_bottles_dustbin
    put_object_cabinet
    rotate_qrcode
    scan_object
    shake_bottle_horizontally
    shake_bottle
    stack_blocks_three
    stack_blocks_two
    stack_bowls_three
    stack_bowls_two
    stamp_seal
    turn_switch
)
total_tasks=${#task_names[@]}

if [ $total_tasks -eq 0 ]; then
    echo "❌ Error: No tasks provided!"
    exit 1
fi

echo "✅ Total tasks: $total_tasks"
echo "✅ Max concurrent: $MAX_CONCURRENT ($NUM_GPUS GPUs × $TASKS_PER_GPU tasks/GPU)"

# 创建日志目录
LOG_DIR="${log_path}/logs/${task_config}"
mkdir -p "$LOG_DIR"

# === 任务执行函数 ===
run_task() {
    local task="$1"
    local gpu_id="$2"
    local port="$3"
    local log_file="${LOG_DIR}/${task}.log"
    
    echo "▶️ GPU $gpu_id: starting [$task] (port=$port, log=$log_file)"
    
    CUDA_VISIBLE_DEVICES="$gpu_id" \
    PYTHONWARNINGS=ignore::UserWarning \
    python script/eval_policy.py --config "$DEPLOY_POLICY_PATH" \
        --overrides \
        task_name "$task" \
        task_config "$task_config" \
        seed "$seed" \
        policy_name "$policy_name" \
        port "$port" \
        policy_ckpt_path "$policy_ckpt_path" \
        log_path "$log_path" \
        2>&1 | tee "$log_file"
}

# 导出函数供 xargs 使用
export -f run_task
export ROBOTWIN_PATH EVAL_FILES_PATH DEPLOY_POLICY_PATH
export policy_ckpt_path log_path task_config seed policy_name LOG_DIR

# === 构建任务列表（task,gpu_id,port）===
# 使用轮询分配 GPU: task0→GPU0, task1→GPU1, ..., task8→GPU0...
{
    for i in "${!task_names[@]}"; do
        gpu_id=$((i % NUM_GPUS))
        port=$((5695 + i))
        echo "${task_names[i]},$gpu_id,$port"
    done
} | xargs -n 3 -P "$MAX_CONCURRENT" -I {} bash -c '
    IFS="," read -r task gpu_id port <<< "{}"
    run_task "$task" "$gpu_id" "$port"
'

echo "✅ All tasks completed!"