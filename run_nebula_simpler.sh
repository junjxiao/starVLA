#!/bin/bash

# 定义任务列表（可以直接注释掉部份任务）
task_names=(
    PutEggplantInBasketScene-v0
    StackGreenCubeOnYellowCubeBakedTexInScene-v0
    PutCarrotOnPlateInScene-v0
    PutSpoonOnTableClothInScene-v0
)

# 过滤掉注释行和空行（只保留非注释的有效任务）
valid_tasks=()
for task in "${task_names[@]}"; do
    # 跳过以 # 开头或空行
    if [[ -n "$task" && "$task" != "#"* ]]; then
        valid_tasks+=("$task")
    fi
done

echo "Found ${#valid_tasks[@]} valid tasks:"
printf '  - %s\n' "${valid_tasks[@]}"

# 每 4 个任务一组，调用 eval.sh
group_size=4
total_tasks=${#valid_tasks[@]}

for ((i=0; i<total_tasks; i+=group_size)); do
    # 提取当前组（最多 group_size 个）
    group=("${valid_tasks[@]:i:group_size}")
    
    echo "Running eval.sh with tasks: ${group[*]}"
    
    # 调用 eval.sh 并传入当前组的任务作为参数
    nebulactl run mdl --queue=amap_app_common_h20_na175 \
                  --entry="bash ./examples/SimplerEnv/eval_files/eval_nebula/eval.sh ${group[*]}"\
                  --user_params="" \
                  --worker_count=1 \
                  --algoame=pytorch260\
                  --file.cluster_file=./cluster.json \
                  --job_name="simpler" \
                  --nas_file_system_id=92bcb4b594-nvt70.cn-zhangjiakou.nas.aliyuncs.com,29016449f1c-mkq60.cn-wulanchabu.nas.aliyuncs.com,9dc4e499f2-tek11.cn-zhangjiakou.nas.aliyuncs.com,29e2cf482cb-cxw73.cn-wulanchabu.nas.aliyuncs.com \
                  --nas_file_system_mount_path=/mnt/workspace,/mnt/nas-data-3,/mnt/nas-data-1,/mnt/xlab-nas-1 \
                  --custom_docker_image=hub.docker.alibaba-inc.com/mdl/notebook_saved:xiaojunjin.xjj_simpler2_20260201014810

done

echo "All groups processed successfully!"
