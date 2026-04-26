import os
import torch
import torchvision
import numpy as np
from PIL import Image
import gc

# 设置视频后端为 pyav (推荐，比 video_reader 更稳定)
try:
    torchvision.set_video_backend("pyav")
except Exception:
    pass # 某些新版本 torchvision 可能不再需要或支持此设置，忽略错误

def extract_middle_frame(video_path, output_path):
    """
    使用 torchvision (pyav backend) 读取视频的中间帧并保存为图片
    
    Args:
        video_path (str): 输入视频路径
        output_path (str): 输出图片路径
        
    Returns:
        bool: 是否成功保存
    """
    if not os.path.exists(video_path):
        print(f"错误: 文件不存在 - {video_path}")
        return False

    reader = None
    try:
        # 1. 初始化 VideoReader
        reader = torchvision.io.VideoReader(video_path, "video")
        
        # 2. 获取元数据以计算中间时间点
        metadata = reader.get_metadata()
        
        if 'video' not in metadata or 'duration' not in metadata['video']:
            print(f"错误: 无法获取视频时长元数据 - {video_path}")
            return False
            
        # 获取总时长 (秒)
        duration = float(metadata['video']['duration'][0])
        
        if duration <= 0:
            print(f"错误: 视频时长为0或无效 - {video_path}")
            return False

        # 3. 计算中间时间戳
        target_ts = duration / 2.0
        # target_ts = 0.1 
        
        # 4. Seek 到中间时间点附近
        # keyframes_only=False 允许 seek 到非关键帧，但可能需要解码更多数据
        # 通常 seek 会跳到最近的关键帧，然后我们需要向后读取几帧找到最接近 target_ts 的帧
        reader.seek(target_ts, keyframes_only=False)
        
        closest_frame_data = None
        closest_ts_diff = float('inf')
        found_frame = False
        
        # 5. 遍历 seek 之后的帧，找到时间戳最接近 target_ts 的那一帧
        for frame in reader:
            current_ts = frame["pts"]
            current_diff = abs(current_ts - target_ts)
            
            # 如果差异开始变大，说明我们已经过了最接近的点，可以停止
            if current_diff > closest_ts_diff:
                break
                
            # 更新最接近的帧
            if current_diff < closest_ts_diff:
                # 清理前一帧的引用以节省内存
                if closest_frame_data is not None:
                    del closest_frame_data
                closest_ts_diff = current_diff
                closest_frame_data = frame["data"] # Tensor [C, H, W]
                
            # 如果差异非常小（例如小于 1ms），可以提前退出
            if current_diff < 1e-4:
                found_frame = True
                break
                
        if closest_frame_data is None:
            print(f"❌ 无法在视频中定位到中间帧 - {video_path}")
            return False

        # 6. 处理图像数据
        # closest_frame_data 是 Tensor [C, H, W], 范围通常是 [0, 1] (float) 或 [0, 255] (uint8)
        # 确保在 CPU 上
        if closest_frame_data.is_cuda:
            closest_frame_data = closest_frame_data.cpu()
            
        # 转换为 numpy array (H, W, C)
        # torchvision 读取的通常是 RGB
        if closest_frame_data.dtype == torch.float32:
            # 如果是 float [0, 1]，转换为 uint8 [0, 255]
            np_frame = (closest_frame_data.numpy() * 255).astype(np.uint8)
        else:
            # 如果是 uint8
            np_frame = closest_frame_data.numpy()
            
        # Transpose from [C, H, W] to [H, W, C]
        np_frame = np.transpose(np_frame, (1, 2, 0))
        
        # 7. 保存图片
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 使用 PIL 保存，因为它直接支持 RGB numpy array
        img = Image.fromarray(np_frame)
        img.save(output_path)
        
        print(f"✅ 成功保存中间帧: {output_path} (Time: {target_ts:.2f}s)")
        return True

    except Exception as e:
        print(f"❌ 处理视频时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # 8. 彻底清理资源 (参考你的代码风格)
        if reader is not None:
            # 尝试关闭容器
            if hasattr(reader, 'container') and reader.container is not None:
                try:
                    reader.container.close()
                except:
                    pass
                reader.container = None
            
            # 清理内部 C 指针 (如果存在)
            if hasattr(reader, '_c'):
                reader._c = None
                
            # 删除 reader 对象
            del reader
            
        # 强制垃圾回收
        gc.collect()



# def extract_middle_frame(video_path, output_path):
#     """
#     读取视频的中间帧并保存为图片
    
#     Args:
#         video_path (str): 输入视频路径
#         output_path (str): 输出图片路径
#     """
#     # 检查文件是否存在
#     if not os.path.exists(video_path):
#         print(f"错误: 文件不存在 - {video_path}")
#         return False

#     # 打开视频文件
#     cap = cv2.VideoCapture(video_path)
    
#     if not cap.isOpened():
#         print(f"错误: 无法打开视频文件 - {video_path}")
#         return False

#     # 获取视频总帧数
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
#     if total_frames <= 0:
#         print(f"错误: 视频帧数为0或无法读取 - {video_path}")
#         cap.release()
#         return False

#     # 计算中间帧的索引 (从0开始)
#     # middle_frame_idx = total_frames // 2
#     middle_frame_idx = 0
    
#     # 设置视频指针到中间帧
#     cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
    
#     # 读取当前帧
#     ret, frame = cap.read()
    
#     if ret:
#         # 确保输出目录存在
#         output_dir = os.path.dirname(output_path)
#         if output_dir and not os.path.exists(output_dir):
#             os.makedirs(output_dir)
            
#         # 保存图片
#         success = cv2.imwrite(output_path, frame)
#         if success:
#             print(f"✅ 成功保存中间帧: {output_path} (第 {middle_frame_idx + 1}/{total_frames} 帧)")
#         else:
#             print(f"❌ 保存失败: {output_path}")
#         cap.release()
#         return success
#     else:
#         print(f"❌ 无法读取第 {middle_frame_idx} 帧")
#         cap.release()
#         return False

def main():
    # 定义输入视频路径
    video_mid = "/mnt/xlab-nas-1/junjin/dataset/libero_mv_video/libero_spatial/mid/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo13.mp4"
    video_left = "/mnt/xlab-nas-1/junjin/dataset/libero_mv_video/libero_spatial/left/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo13.mp4"
    video_right = "/mnt/xlab-nas-1/junjin/dataset/libero_mv_video/libero_spatial/right/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo13.mp4"

    # 定义输出图片路径 (保存在当前目录下，或者你可以指定其他目录)
    output_mid = "./plots/my_figs/exp/depth/vis3/input.jpg"
    output_left = "./plots/my_figs/exp/depth/vis3/left_frame.jpg"
    output_right = "./plots/my_figs/exp/depth/vis3/right_frame.jpg"

    print("开始处理 mid 视频...")
    extract_middle_frame(video_mid, output_mid)

    print("开始处理 left 视频...")
    extract_middle_frame(video_left, output_left)

    print("开始处理 right 视频...")
    extract_middle_frame(video_right, output_right)

    print("处理完成！")

# if __name__ == "__main__":
#     main()


import h5py
import numpy as np
import cv2

# 打开文件（只读模式）

with h5py.File('/mnt/xlab-nas-1/junjin/dataset/regenerated_libero/libero_spatial_no_noops/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5', 'r') as f:
    # import ipdb
    # ipdb.set_trace()
    l = len(f['data']['demo_13']['obs']['agentview_rgb'])
    # main_view = f['data']['demo_2']['obs']['agentview_rgb'][10]
    # wrist_view = f['data']['demo_2']['obs']['eye_in_hand_rgb'][10]
    depth_val = f['data']['demo_13']['obs']['agentview_depth'][l//2][:,:,0]
    d_min = depth_val.min()
    d_max = depth_val.max()
    
    if d_max - d_min > 1e-8:
        depth_vis = (depth_val - d_min) / (d_max - d_min) * 255.0
    else:
        depth_vis = np.zeros_like(depth_val)
    
    depth_vis = depth_vis.astype(np.uint8)
    depth_vis_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
    cv2.imwrite('./plots/my_figs/exp/depth/vis3/gt.png', depth_vis_colored)
    # cv2.imwrite('./plots/my_figs/method/input.png', main_view[:,:,[2,1,0]])
    # cv2.imwrite('./plots/my_figs/method/wrist_input.png', wrist_view[:,:,[2,1,0]])
