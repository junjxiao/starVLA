import os
import glob
import imageio
import numpy as np
from PIL import Image

# ================= 配置区域 =================
ROOT_DIR = "./videos" 
# ===========================================

def convert_video_via_frames(input_path, output_path):
    """
    1. 读取视频的所有帧到内存 (作为 numpy 数组或 PIL Image)
    2. 保持原始 FPS
    3. 使用 H.264 编码重新保存
    """
    try:
        print(f"⏳ 正在读取帧: {os.path.basename(input_path)} ...")
        
        # 1. 读取视频所有帧
        # reader = imageio.get_reader(input_path)
        # frames = []
        # for frame in reader:
        #     frames.append(frame)
        # fps = reader.get_meta_data()['fps']
        # reader.close()
        
        # 更现代的写法 (imageio v2.9+):
        with imageio.get_reader(input_path) as reader:
            meta = reader.get_meta_data()
            fps = meta.get('fps', 25) # 获取原始 FPS
            frames = [frame for frame in reader] # 将所有帧读入列表
            
        print(f"   ℹ️ 读取完成: {len(frames)} 帧, FPS: {fps}")

        if not frames:
            print("❌ 未读取到任何帧")
            return False

        # 2. 保存视频
        print(f"💾 正在保存: {os.path.basename(output_path)} ...")
        
        # 使用 imageio 写入
        # codec='libx264' 确保浏览器兼容
        # fps=fps 保持原始帧率
        with imageio.get_writer(output_path, fps=fps, codec='libx264') as writer:
            for frame in frames:
                writer.append_data(frame)
                
        print(f"✅ 成功: {output_path}")
        return True

    except Exception as e:
        print(f"❌ 失败: {input_path}")
        print(f"   错误信息: {str(e)}")
        return False

def batch_convert(root_dir):
    if not os.path.exists(root_dir):
        print(f"❌ 目录不存在: {root_dir}")
        return

    print(f"🚀 开始扫描目录: {root_dir}")
    print(f"⚙️ 目标: 读取所有帧 -> 保持原FPS -> 编码=H.264")
    print("-" * 40)

    success_count = 0
    fail_count = 0

    # 支持的视频后缀
    extensions = ['*.mp4', '*.mov', '*.avi', '*.mkv', '*.flv', '*.webm']
    
    for ext in extensions:
        for video_path in glob.glob(os.path.join(root_dir, '**', ext), recursive=True):
            
            # 构造输出路径
            dir_name = os.path.dirname(video_path)
            file_name = os.path.basename(video_path)
            name_without_ext = os.path.splitext(file_name)[0]
            
            # 统一输出为 .mp4
            output_path = os.path.join(dir_name, f"{name_without_ext}.mp4")

            # 执行转换
            if convert_video_via_frames(video_path, output_path):
                success_count += 1
            else:
                fail_count += 1

    print("-" * 40)
    print(f"🎉 处理完成! 成功: {success_count}, 失败: {fail_count}")

if __name__ == "__main__":
    batch_convert(ROOT_DIR)
