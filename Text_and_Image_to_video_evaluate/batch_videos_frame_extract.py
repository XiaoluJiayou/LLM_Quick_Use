import os
import cv2
import glob
import time


def extract_single_video(video_path, save_dir, extract_fps=1):
    """
    对单个视频进行抽帧处理

    Args:
        video_path: 视频文件的完整路径
        save_dir: 保存图片的文件夹路径
        extract_fps: 抽帧率 (每秒抽取多少帧)

    Returns:
        dict: 包含视频信息和处理结果的字典，如果失败则返回 None
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    # 获取视频基本信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    # 容错处理：无法读取帧率
    if video_fps == 0:
        cap.release()
        return None

    # 计算抽帧间隔
    if extract_fps is None or extract_fps >= video_fps:
        frame_interval = 1
    else:
        frame_interval = int(round(video_fps / extract_fps))

    frame_count = 0
    saved_count = 0

    # 开始抽帧循环
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 抽帧循环里，保存图片前加一个判断
        if saved_count >= 30:  # 最多只抽30帧，防止后续评估显存爆炸
            break

        # 符合间隔则保存
        if frame_count % frame_interval == 0:
            # # 修改抽帧后图片的保存路劲名
            # save_name = f"images_{saved_count + 1}.jpg"
            # 修改后，直接从001开始，如：001.jpg, 002.jpg，利于后续评估代码的中的路劲排序
            save_name = f"{saved_count + 1:03d}.jpg"
            save_path = os.path.join(save_dir, save_name)
            cv2.imwrite(save_path, frame)
            saved_count += 1

        frame_count += 1

    cap.release()

    # 返回处理结果统计数据
    return {
        'total_frames': total_frames,
        'video_fps': video_fps,
        'saved_count': saved_count,
        'duration': total_frames / video_fps if video_fps > 0 else 0,
        'interval': frame_interval
    }


def batch_extract_frames(input_dir='Videos', output_dir='Videos-Frames', extract_fps=1):
    """
    批量处理文件夹下的所有视频
    """
    # 支持的格式
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv', '*.MP4', '*.AVI']

    # 1. 输入检查
    if not os.path.exists(input_dir):
        print(f"错误：未找到文件夹 '{input_dir}'")
        return

    # 2. 搜索视频文件
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(input_dir, ext)))

    if not video_files:
        print(f"在 '{input_dir}' 下未找到视频文件。")
        return

    print(f"--- 任务开始 --- 共发现 {len(video_files)} 个视频 ---\n")

    # 创建输出根目录
    os.makedirs(output_dir, exist_ok=True)

    # 3. 遍历并调用抽帧函数
    for video_path in video_files:
        video_name_with_ext = os.path.basename(video_path)
        video_name = os.path.splitext(video_name_with_ext)[0]

        # 构建保存路径：Videos-Frames -> Video_Name_Frames
        video_save_dir = os.path.join(output_dir, video_name + "_Frames")
        os.makedirs(video_save_dir, exist_ok=True)

        print(f"正在处理: {video_name_with_ext}")

        # 调用核心抽帧函数
        start_time = time.time()
        result = extract_single_video(video_path, video_save_dir, extract_fps)
        end_time = time.time()

        # 打印统计信息
        if result:
            print(f"  [信息] 原始帧率: {result['video_fps']:.2f} FPS | 时长: {result['duration']:.2f}秒")
            print(f"  [信息] 抽帧间隔: 每 {result['interval']} 帧 | 抽帧率: {extract_fps} FPS")
            print(f"  [完成] 保存数量: {result['saved_count']} 张 | 耗时: {end_time - start_time:.2f}秒")
            print(f"  [路径] {video_save_dir}\n" + "-" * 50)
        else:
            print(f"  [失败] 无法读取视频或帧率错误。\n" + "-" * 50)

    print("所有视频处理完毕！")


if __name__ == "__main__":
    # === 参数配置 ===
    INPUT_FOLDER = r'./Videos'
    OUTPUT_FOLDER = r'Videos-Frames'
    EXTRACT_FPS = 1  # 每秒抽取1帧

    batch_extract_frames(INPUT_FOLDER, OUTPUT_FOLDER, EXTRACT_FPS)
