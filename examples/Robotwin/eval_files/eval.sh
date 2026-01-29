#!/bin/bash
export CUROBO_TORCH_COMPILE_DISABLE=1
port=5695
gpu_id=2
policy_ckpt_path=/mnt/workspace/junjin/code/starVLA/checkpoints/0127_robotwin_Qwen3vlGR00T_vggt_cross_bs16/checkpoints/steps_10000_pytorch_model.pt
ckpt_setting="0127_robotwin_Qwen3vlGR00T_vggt_cross_bs16_step10000"
log_path=/mnt/workspace/junjin/code/starVLA/outputs/robotwin/

seed=0
task_config="demo_clean" # demo_randomized

ROBOTWIN_PATH=/mnt/workspace/junjin/code/RoboTwin
policy_name="model2robotwin_interface"


export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

EVAL_FILES_PATH=/mnt/workspace/junjin/code/starVLA/examples/Robotwin/eval_files
STARVLA_PATH=/mnt/workspace/junjin/code/starVLA
DEPLOY_POLICY_PATH=$EVAL_FILES_PATH/deploy_policy.yml

export PYTHONPATH=$ROBOTWIN_PATH:$PYTHONPATH
export PYTHONPATH=$STARVLA_PATH:$PYTHONPATH
export PYTHONPATH=$EVAL_FILES_PATH:$PYTHONPATH

cd $ROBOTWIN_PATH

echo "PYTHONPATH: $PYTHONPATH"

PYTHONWARNINGS=ignore::UserWarning \
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
for i in "${!task_names[@]}"; do
    python script/eval_policy.py --config $DEPLOY_POLICY_PATH \
        --overrides \
        task_name ${task_names[$i]} \
        task_config ${task_config} \
        ckpt_setting ${ckpt_setting} \
        seed ${seed} \
        policy_name ${policy_name} \
        port ${port} \
        policy_ckpt_path ${policy_ckpt_path} \
        log_path ${log_path}
done