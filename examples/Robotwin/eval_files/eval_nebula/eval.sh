#!/bin/bash
eval "$(conda shell.bash hook)"
# source activate
conda activate python3.10

task_names=("$@")

export CUROBO_TORCH_COMPILE_DISABLE=1
port=5695
gpu_id=0
policy_ckpt_path=/mnt/workspace/junjin/code/starVLA/checkpoints/0127_robotwin_Qwen3vlGR00T_vggt_cross_bs16/checkpoints/steps_10000_pytorch_model.pt
log_path=/mnt/workspace/junjin/code/starVLA/outputs/robotwin/0127_robotwin_Qwen3vlGR00T_vggt_cross_bs16_step10000

mkdir -p ${LOG_DIR}
seed=0
task_config="demo_clean" # demo_randomized
LOG_DIR="${log_path}/logs/${task_config}"
ROBOTWIN_PATH=/mnt/workspace/junjin/code/RoboTwin
policy_name="model2robotwin_interface"


export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

EVAL_FILES_PATH=/mnt/workspace/junjin/code/starVLA/examples/Robotwin/eval_files/eval_nebula/
STARVLA_PATH=/mnt/workspace/junjin/code/starVLA
DEPLOY_POLICY_PATH=$EVAL_FILES_PATH/deploy_policy.yml

export PYTHONPATH=$ROBOTWIN_PATH:$PYTHONPATH
export PYTHONPATH=$STARVLA_PATH:$PYTHONPATH
export PYTHONPATH=$EVAL_FILES_PATH:$PYTHONPATH

cd $ROBOTWIN_PATH

echo "PYTHONPATH: $PYTHONPATH"



# ckpt_setting ${ckpt_setting} \
for i in "${!task_names[@]}"; do
    log_file="${LOG_DIR}/${task_names[$i]}.log"
    PYTHONWARNINGS=ignore::UserWarning \
    python script/eval_policy.py --config $DEPLOY_POLICY_PATH \
        --overrides \
        task_name ${task_names[$i]} \
        task_config ${task_config} \
        seed ${seed} \
        policy_name ${policy_name} \
        port ${port} \
        policy_ckpt_path ${policy_ckpt_path} \
        log_path ${log_path} \
        2>&1 | tee "${log_file}" &
done

wait