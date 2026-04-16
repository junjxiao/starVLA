import cv2
import os

def extract_middle_frame(video_path, output_path):
    """
    读取视频的中间帧并保存为图片
    
    Args:
        video_path (str): 输入视频路径
        output_path (str): 输出图片路径
    """
    # 检查文件是否存在
    if not os.path.exists(video_path):
        print(f"错误: 文件不存在 - {video_path}")
        return False

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 - {video_path}")
        return False

    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        print(f"错误: 视频帧数为0或无法读取 - {video_path}")
        cap.release()
        return False

    # 计算中间帧的索引 (从0开始)
    middle_frame_idx = total_frames // 2
    
    # 设置视频指针到中间帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
    
    # 读取当前帧
    ret, frame = cap.read()
    
    if ret:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 保存图片
        success = cv2.imwrite(output_path, frame)
        if success:
            print(f"✅ 成功保存中间帧: {output_path} (第 {middle_frame_idx + 1}/{total_frames} 帧)")
        else:
            print(f"❌ 保存失败: {output_path}")
        cap.release()
        return success
    else:
        print(f"❌ 无法读取第 {middle_frame_idx} 帧")
        cap.release()
        return False

def main():
    # 定义输入视频路径
    video_mid = "/mnt/xlab-nas-1/junjin/dataset/libero_mv_video/libero_spatial/mid/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_demo5.mp4"
    video_left = "/mnt/xlab-nas-1/junjin/dataset/libero_mv_video/libero_spatial/left/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_demo5.mp4"
    video_right = "/mnt/xlab-nas-1/junjin/dataset/libero_mv_video/libero_spatial/right/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_demo5.mp4"

    # 定义输出图片路径 (保存在当前目录下，或者你可以指定其他目录)
    output_mid = "./assets/my_figs/mv_imgs/mid_frame.jpg"
    output_left = "./assets/my_figs/mv_imgs/left_frame.jpg"
    output_right = "./assets/my_figs/mv_imgs/right_frame.jpg"

    print("开始处理 mid 视频...")
    extract_middle_frame(video_mid, output_mid)

    print("开始处理 left 视频...")
    extract_middle_frame(video_left, output_left)

    print("开始处理 right 视频...")
    extract_middle_frame(video_right, output_right)

    print("处理完成！")

if __name__ == "__main__":
    main()
