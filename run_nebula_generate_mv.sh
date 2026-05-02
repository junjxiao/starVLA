# #!/bin/bash

# # 定义数组

# tasks=(
#     "RoboTwin-Clean/adjust_bottle"
#     "RoboTwin-Clean/beat_block_hammer"
#     "RoboTwin-Clean/blocks_ranking_rgb"
#     "RoboTwin-Clean/blocks_ranking_size"
#     "RoboTwin-Clean/click_alarmclock"
#     "RoboTwin-Clean/click_bell"
#     "RoboTwin-Clean/dump_bin_bigbin"
#     "RoboTwin-Clean/grab_roller"
#     "RoboTwin-Clean/handover_block"
#     "RoboTwin-Clean/handover_mic"
#     "RoboTwin-Clean/hanging_mug"
#     "RoboTwin-Clean/lift_pot"
#     "RoboTwin-Clean/move_can_pot"
#     "RoboTwin-Clean/move_pillbottle_pad"
#     "RoboTwin-Clean/move_playingcard_away"
#     "RoboTwin-Clean/move_stapler_pad"
#     "RoboTwin-Clean/open_laptop"
#     "RoboTwin-Clean/open_microwave"
#     "RoboTwin-Clean/pick_diverse_bottles"
#     "RoboTwin-Clean/pick_dual_bottles"
#     "RoboTwin-Clean/place_a2b_left"
#     "RoboTwin-Clean/place_a2b_right"
#     "RoboTwin-Clean/place_bread_basket"
#     "RoboTwin-Clean/place_bread_skillet"
#     "RoboTwin-Clean/place_burger_fries"
#     "RoboTwin-Clean/place_can_basket"
#     "RoboTwin-Clean/place_cans_plasticbox"
#     "RoboTwin-Clean/place_container_plate"
#     "RoboTwin-Clean/place_dual_shoes"
#     "RoboTwin-Clean/place_empty_cup"
#     "RoboTwin-Clean/place_fan"
#     "RoboTwin-Clean/place_mouse_pad"
#     "RoboTwin-Clean/place_object_basket"
#     "RoboTwin-Clean/place_object_scale"
#     "RoboTwin-Clean/place_object_stand"
#     "RoboTwin-Clean/place_phone_stand"
#     "RoboTwin-Clean/place_shoe"
#     "RoboTwin-Clean/press_stapler"
#     "RoboTwin-Clean/put_bottles_dustbin"
#     "RoboTwin-Clean/put_object_cabinet"
#     "RoboTwin-Clean/rotate_qrcode"
#     "RoboTwin-Clean/scan_object"
#     "RoboTwin-Clean/shake_bottle"
#     "RoboTwin-Clean/shake_bottle_horizontally"
#     "RoboTwin-Clean/stack_blocks_three"
#     "RoboTwin-Clean/stack_blocks_two"
#     "RoboTwin-Clean/stack_bowls_three"
#     "RoboTwin-Clean/stack_bowls_two"
#     "RoboTwin-Clean/stamp_seal"
#     "RoboTwin-Clean/turn_switch"
#     "RoboTwin-convert/lerobot/adjust_bottle"
#     "RoboTwin-convert/lerobot/beat_block_hammer"
#     "RoboTwin-convert/lerobot/blocks_ranking_rgb"
#     "RoboTwin-convert/lerobot/blocks_ranking_size"
#     "RoboTwin-convert/lerobot/click_alarmclock"
#     "RoboTwin-convert/lerobot/click_bell"
#     "RoboTwin-convert/lerobot/dump_bin_bigbin"
#     "RoboTwin-convert/lerobot/grab_roller"
#     "RoboTwin-convert/lerobot/handover_block"
#     "RoboTwin-convert/lerobot/handover_mic"
#     "RoboTwin-convert/lerobot/hanging_mug"
#     "RoboTwin-convert/lerobot/lift_pot"
#     "RoboTwin-convert/lerobot/move_can_pot"
#     "RoboTwin-convert/lerobot/move_pillbottle_pad"
#     "RoboTwin-convert/lerobot/move_playingcard_away"
#     "RoboTwin-convert/lerobot/move_stapler_pad"
#     "RoboTwin-convert/lerobot/open_laptop"
#     "RoboTwin-convert/lerobot/open_microwave"
#     "RoboTwin-convert/lerobot/pick_diverse_bottles"
#     "RoboTwin-convert/lerobot/pick_dual_bottles"
#     "RoboTwin-convert/lerobot/place_a2b_left"
#     "RoboTwin-convert/lerobot/place_a2b_right"
#     "RoboTwin-convert/lerobot/place_bread_basket"
#     "RoboTwin-convert/lerobot/place_bread_skillet"
#     "RoboTwin-convert/lerobot/place_burger_fries"
#     "RoboTwin-convert/lerobot/place_can_basket"
#     "RoboTwin-convert/lerobot/place_cans_plasticbox"
#     "RoboTwin-convert/lerobot/place_container_plate"
#     "RoboTwin-convert/lerobot/place_dual_shoes"
#     "RoboTwin-convert/lerobot/place_empty_cup"
#     "RoboTwin-convert/lerobot/place_fan"
#     "RoboTwin-convert/lerobot/place_mouse_pad"
#     "RoboTwin-convert/lerobot/place_object_basket"
#     "RoboTwin-convert/lerobot/place_object_scale"
#     "RoboTwin-convert/lerobot/place_object_stand"
#     "RoboTwin-convert/lerobot/place_phone_stand"
#     "RoboTwin-convert/lerobot/place_shoe"
#     "RoboTwin-convert/lerobot/press_stapler"
#     "RoboTwin-convert/lerobot/put_bottles_dustbin"
#     "RoboTwin-convert/lerobot/put_object_cabinet"
#     "RoboTwin-convert/lerobot/rotate_qrcode"
#     "RoboTwin-convert/lerobot/scan_object"
#     "RoboTwin-convert/lerobot/shake_bottle"
#     "RoboTwin-convert/lerobot/shake_bottle_horizontally"
#     "RoboTwin-convert/lerobot/stack_blocks_three"
#     "RoboTwin-convert/lerobot/stack_blocks_two"
#     "RoboTwin-convert/lerobot/stack_bowls_three"
#     "RoboTwin-convert/lerobot/stack_bowls_two"
#     "RoboTwin-convert/lerobot/stamp_seal"
#     "RoboTwin-convert/lerobot/turn_switch"
# )

# # sizes=(379 428 454 432)
# # tasks=("libero_10_no_noops_1.0.0_lerobot" "libero_goal_no_noops_1.0.0_lerobot" "libero_object_no_noops_1.0.0_lerobot" "libero_spatial_no_noops_1.0.0_lerobot")
# # total_slices=(4 4 4 4)

# # for i in "${!sizes[@]}"; do
# for i in "${!tasks[@]}"; do
#     # size=${sizes[$i]}
#     size=50
#     task=${tasks[$i]}
#     # num_slice=${total_slices[$i]}
#     num_slice=1
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
#         nebulactl run mdl --queue=amap-poi_ppu810e \
#                   --entry="bash generate_mv.sh $task $start_idx $end_idx"\
#                   --user_params="" \
#                   --worker_count=1 \
#                   --algoame=pytorch260\
#                   --file.cluster_file=./cluster.json \
#                   --job_name="robotwin" \
#                   --nas_file_system_id=92bcb4b594-nvt70.cn-zhangjiakou.nas.aliyuncs.com,29016449f1c-mkq60.cn-wulanchabu.nas.aliyuncs.com,9dc4e499f2-tek11.cn-zhangjiakou.nas.aliyuncs.com,29e2cf482cb-cxw73.cn-wulanchabu.nas.aliyuncs.com \
#                   --nas_file_system_mount_path=/mnt/workspace,/mnt/nas-data-3,/mnt/nas-data-1,/mnt/xlab-nas-1 \
#                   --custom_docker_image=hub.docker.alibaba-inc.com/mdl/notebook_saved:xiaojunjin.xjj_libero_plus_ppu2_20260119180727
#         start_idx=$end_idx
#     done
# done


# # nebulactl run mdl --queue=amap-poi_ppu810e \
# #                   --entry="bash examples/LIBERO-plus/eval_files/eval_nebula/eval_libero_in_one.sh libero_goal 1296 1943"\
# #                   --user_params="" \
# #                   --worker_count=1 \
# #                   --algoame=pytorch260\
# #                   --file.cluster_file=./cluster.json \
# #                   --job_name="libero_plus" \
# #                   --nas_file_system_id=92bcb4b594-nvt70.cn-zhangjiakou.nas.aliyuncs.com,29016449f1c-mkq60.cn-wulanchabu.nas.aliyuncs.com,9dc4e499f2-tek11.cn-zhangjiakou.nas.aliyuncs.com,29e2cf482cb-cxw73.cn-wulanchabu.nas.aliyuncs.com \
# #                   --nas_file_system_mount_path=/mnt/workspace,/mnt/nas-data-3,/mnt/nas-data-1,/mnt/xlab-nas-1 \
# #                   --custom_docker_image=hub.docker.alibaba-inc.com/mdl/notebook_saved:xiaojunjin.xjj_libero_plus_ppu2_20260119180727
# # nebulactl run mdl --queue=amap-poi_ppu810e \
# #                   --entry="bash examples/LIBERO-plus/eval_files/eval_nebula/eval_libero_in_one.sh libero_goal 1943 2591"\
# #                   --user_params="" \
# #                   --worker_count=1 \
# #                   --algoame=pytorch260\
# #                   --file.cluster_file=./cluster.json \
# #                   --job_name="libero_plus" \
# #                   --nas_file_system_id=92bcb4b594-nvt70.cn-zhangjiakou.nas.aliyuncs.com,29016449f1c-mkq60.cn-wulanchabu.nas.aliyuncs.com,9dc4e499f2-tek11.cn-zhangjiakou.nas.aliyuncs.com,29e2cf482cb-cxw73.cn-wulanchabu.nas.aliyuncs.com \
# #                   --nas_file_system_mount_path=/mnt/workspace,/mnt/nas-data-3,/mnt/nas-data-1,/mnt/xlab-nas-1 \
# #                   --custom_docker_image=hub.docker.alibaba-inc.com/mdl/notebook_saved:xiaojunjin.xjj_libero_plus_ppu2_20260119180727

# # amap-poi_ppu810e   amap_app_common_h20_na175
# # nebulactl run mdl --queue=amap-poi_ppu810e \
# #                   --entry="bash examples/LIBERO-plus/eval_files/eval_nebula/eval_libero_in_one.sh libero_goal 0 5"\
# #                   --user_params="" \
# #                   --worker_count=1 \
# #                   --algoame=pytorch260\
# #                   --file.cluster_file=./cluster.json \
# #                   --job_name="libero_plus" \
# #                   --nas_file_system_id=92bcb4b594-nvt70.cn-zhangjiakou.nas.aliyuncs.com,29016449f1c-mkq60.cn-wulanchabu.nas.aliyuncs.com,9dc4e499f2-tek11.cn-zhangjiakou.nas.aliyuncs.com,29e2cf482cb-cxw73.cn-wulanchabu.nas.aliyuncs.com \
# #                   --nas_file_system_mount_path=/mnt/workspace,/mnt/nas-data-3,/mnt/nas-data-1,/mnt/xlab-nas-1 \
# #                   --custom_docker_image=hub.docker.alibaba-inc.com/mdl/notebook_saved:xiaojunjin.xjj_libero_plus_ppu2_20260119180727

#!/bin/bash

# 定义基础数据集路径和名称
DATASET_BASE_NAME="bridge_orig_lerobot"
# 原始视频根目录 (根据实际挂载路径调整)
ORIG_ROOT="/mnt/xlab-nas-2/vla_dataset/lerobot/oxe/${DATASET_BASE_NAME}/videos"

# 获取所有 chunk 目录
# 假设目录结构为: .../videos/chunk-000, .../videos/chunk-001
chunks=()
if [ -d "$ORIG_ROOT" ]; then
    for d in "$ORIG_ROOT"/chunk-*; do
        if [ -d "$d" ]; then
            # 只取目录名，例如 chunk-000
            chunks+=($(basename "$d"))
        fi
    done
else
    echo "Error: Directory $ORIG_ROOT does not exist."
    exit 1
fi

echo "Found ${#chunks[@]} chunks to process."

# 遍历每个 chunk 提交任务
for chunk_name in "${chunks[@]}"; do
    echo "Submitting job for chunk: $chunk_name"
    
    # 这里假设每个 chunk 内部视频数量适中，一次性处理完。
    # 如果单个 chunk 视频极多（例如 >1000），可以在这里进一步拆分 start/end
    # 目前设定 start=0, end=-1 (在 python 中解释为处理该 chunk 下所有视频)
    
    nebulactl run mdl --queue=amap-poi_ppu810e \
              --entry="bash generate_mv.sh ${DATASET_BASE_NAME} ${chunk_name} 0 -1" \
              --user_params="" \
              --worker_count=1 \
              --algoame=pytorch260 \
              --file.cluster_file=./cluster.json \
              --job_name="robotwin_mv_${chunk_name}" \
              --nas_file_system_id=92bcb4b594-nvt70.cn-zhangjiakou.nas.aliyuncs.com,29016449f1c-mkq60.cn-wulanchabu.nas.aliyuncs.com,9dc4e499f2-tek11.cn-zhangjiakou.nas.aliyuncs.com,29e2cf482cb-cxw73.cn-wulanchabu.nas.aliyuncs.com \
              --nas_file_system_mount_path=/mnt/workspace,/mnt/nas-data-3,/mnt/nas-data-1,/mnt/xlab-nas-1 \
              --custom_docker_image=hub.docker.alibaba-inc.com/mdl/notebook_saved:xiaojunjin.xjj_libero_plus_ppu2_20260119180727
    
    # 可选：添加短暂睡眠避免瞬间提交过多任务导致队列拥堵
    # sleep 1
done
