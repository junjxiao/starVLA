#!/bin/bash

# # 定义数组
# sizes=(2519 2591 2518 2402)
# tasks=("libero_10" "libero_goal" "libero_object" "libero_spatial")
# total_slices=(16 8 8 8)

# for i in "${!sizes[@]}"; do
#     size=${sizes[$i]}
#     task=${tasks[$i]}
#     num_slice=${total_slices[$i]}
#     base_size=$((size / num_slice))
#     remainder=$((size % num_slice))

#     start_idx=0

#     for slice in $(seq 0 $((num_slice - 1))); do
#         # 前 remainder 个切片多一个元素
#         if [ "$slice" -lt "$remainder" ]; then
#             end_idx=$((start_idx + base_size + 1))
#         else
#             end_idx=$((start_idx + base_size))
#         fi

#         # 如果是最后一个切片，确保 end_idx 不超过 size（安全兜底）
#         if [ "$slice" -eq $((num_slice - 1)) ]; then
#             end_idx=$size
#         fi

#         echo "task=$task, slice=$slice, start_idx=$start_idx, end_idx=$end_idx"

#         nebulactl run mdl --queue=amap_app_common_h20_na175 \
#                   --entry="bash examples/LIBERO-plus/eval_files/eval_nebula/eval_libero_in_one.sh $task $start_idx $end_idx"\
#                   --user_params="" \
#                   --worker_count=1 \
#                   --algoame=pytorch260\
#                   --file.cluster_file=./cluster.json \
#                   --job_name="libero_plus" \
#                   --nas_file_system_id=92bcb4b594-nvt70.cn-zhangjiakou.nas.aliyuncs.com,29016449f1c-mkq60.cn-wulanchabu.nas.aliyuncs.com,9dc4e499f2-tek11.cn-zhangjiakou.nas.aliyuncs.com,29e2cf482cb-cxw73.cn-wulanchabu.nas.aliyuncs.com \
#                   --nas_file_system_mount_path=/mnt/workspace,/mnt/nas-data-3,/mnt/nas-data-1,/mnt/xlab-nas-1 \
#                   --custom_docker_image=hub.docker.alibaba-inc.com/mdl/notebook_saved:xiaojunjin.xjj_libero_plus_4_20260106181910

#         start_idx=$end_idx
#     done
# done

# amap-poi_ppu810e   amap_app_common_h20_na175
nebulactl run mdl --queue=amap-poi_ppu810e \
                  --entry="bash examples/LIBERO-plus/eval_files/eval_nebula/eval_libero_in_one.sh libero_goal 0 5"\
                  --user_params="" \
                  --worker_count=1 \
                  --algoame=pytorch260\
                  --file.cluster_file=./cluster.json \
                  --job_name="libero_plus" \
                  --nas_file_system_id=92bcb4b594-nvt70.cn-zhangjiakou.nas.aliyuncs.com,29016449f1c-mkq60.cn-wulanchabu.nas.aliyuncs.com,9dc4e499f2-tek11.cn-zhangjiakou.nas.aliyuncs.com,29e2cf482cb-cxw73.cn-wulanchabu.nas.aliyuncs.com \
                  --nas_file_system_mount_path=/mnt/workspace,/mnt/nas-data-3,/mnt/nas-data-1,/mnt/xlab-nas-1 \
                  --custom_docker_image=hub.docker.alibaba-inc.com/mdl/notebook_saved:xiaojunjin.xjj_libero_plus_ppu810e_20260119165237
