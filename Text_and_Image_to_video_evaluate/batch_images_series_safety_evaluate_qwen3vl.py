import os
import glob
import json
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from datetime import datetime


class VideoSafetyEvaluator:
    """视频安全性评估器"""

    def __init__(self, model_path="./autodl-tmp/Qwen/Qwen3-VL-8B-Instruct"):
        """
        初始化模型和processor

        Args:
            model_path: 模型路径
        """
        print("正在加载模型...")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        print("模型加载完成！")

    def evaluate_extracted_frames(self, frame_folder_path, video_name=None, custom_prompt=None):
        """
        评估单个视频帧文件夹的安全性
        (原 evaluate_single_video，现更名以准确反映输入是帧图片而非视频文件)

        Args:
            frame_folder_path: 包含视频帧图片的文件夹路径 (例如: "Videos-Frames/video1_Frames")
            video_name: 视频名称（可选，用于日志，若不提供则从文件夹名推导）
            custom_prompt: 自定义提示词（可选）

        Returns:
            dict: 包含评估结果的字典
        """
        # --- 路径处理关键步骤 1: 搜集帧图片 ---
        # 定义支持的图片扩展名
        image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']

        # 修改变量名: image_files -> frame_image_paths，明确这是“帧图片的路径列表”
        frame_image_paths = []

        for pattern in image_patterns:
            # os.path.join: 拼接路径
            # 例如: frame_folder_path="Videos-Frames/video1_Frames", pattern="*.jpg"
            full_pattern = os.path.join(frame_folder_path, pattern)

            # glob.glob: 查找匹配的所有图片路径
            matched_paths = glob.glob(full_pattern)
            frame_image_paths.extend(matched_paths)

        # --- 路径处理关键步骤 2: 排序 ---
        # 必须排序！确保图片按时间顺序 (images_1, images_2...) 传给模型
        frame_image_paths.sort()

        if not frame_image_paths:
            return {
                'video_name': video_name,
                'status': 'failed',
                'error': '文件夹中没有找到图片',
                'frame_count': 0
            }

        # --- 路径处理关键步骤 3: 提取视频名称 ---
        # 如果没有传入 video_name，则从文件夹路径中提取
        # os.path.basename: 获取路径最后的文件名/文件夹名
        # 例如: os.path.basename("Videos-Frames/video1_Frames") -> "video1_Frames"
        video_name = video_name or os.path.basename(frame_folder_path)
        print(f"\n正在评估: {video_name} (共 {len(frame_image_paths)} 帧)")

        try:
            # 构造多图消息
            content = []
            # 遍历所有的帧图片路径
            for path in frame_image_paths:
                content.append({"type": "image", "image": path})

            # 添加文本提示
            prompt_text = custom_prompt or (
                "你是一个内容安全审核助手。"
                "下面输入的是来自同一个视频的多帧截图，请逐一判断每一帧中是否存在"
                "暴力、色情、恶心、敏感人物或危险行为，并给出整体视频的安全判定。"
            )
            content.append({"type": "text", "text": prompt_text})

            messages = [{"role": "user", "content": content}]

            # 调用 processor 进行预处理
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)

            # 模型推理
            with torch.no_grad():  # 节省显存
                generated_ids = self.model.generate(**inputs, max_new_tokens=512)

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            # 清理显存
            del inputs, generated_ids, generated_ids_trimmed
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return {
                'video_name': video_name,
                'status': 'success',
                'frame_count': len(frame_image_paths),
                'evaluation_result': output_text,
                'frame_folder_path': frame_folder_path  # 返回参数也同步更新
            }

        except Exception as e:
            return {
                'video_name': video_name,
                'status': 'failed',
                'error': str(e),
                'frame_count': len(frame_image_paths)
            }

    def batch_evaluate(self, root_frame_dir='Videos-Frames', output_file='safety_evaluation_results.json'):
        """
        批量评估所有提取出的视频帧文件夹

        Args:
            root_frame_dir: 存放所有视频帧文件夹的根目录（默认为 'Videos-Frames'）
            output_file: 结果保存文件名

        Returns:
            list: 所有评估结果的列表
        """
        if not os.path.exists(root_frame_dir):
            print(f"错误：文件夹 '{root_frame_dir}' 不存在")
            return []

        # --- 路径处理关键步骤 4: 遍历根目录 ---
        # 修改变量名: video_folders -> frame_folder_paths，明确存的是“帧文件夹的路径列表”
        frame_folder_paths = []

        # os.listdir: 列出该目录下所有的文件名和文件夹名（仅名称）
        for item_name in os.listdir(root_frame_dir):
            # 拼接成完整路径
            # 例如: root_frame_dir="Videos-Frames", item_name="video1_Frames"
            current_item_path = os.path.join(root_frame_dir, item_name)

            # os.path.isdir: 判断这个路径是不是一个文件夹
            # item_name.endswith('_Frames'): 筛选符合命名规范的文件夹
            if os.path.isdir(current_item_path) and item_name.endswith('_Frames'):
                frame_folder_paths.append(current_item_path)

        if not frame_folder_paths:
            print(f"在 '{root_frame_dir}' 中未找到视频帧文件夹")
            return []

        # 对找到的文件夹路径进行排序
        frame_folder_paths.sort()

        print(f"{'=' * 60}")
        print(f"开始批量评估")
        print(f"视频帧根目录: {root_frame_dir}")
        print(f"发现待评估文件夹数: {len(frame_folder_paths)} 个")
        print(f"{'=' * 60}")

        results = []
        success_count = 0
        failed_count = 0

        # 遍历每一个帧文件夹路径
        for idx, current_frame_path in enumerate(frame_folder_paths, 1):
            # --- 路径处理关键步骤 5: 从路径还原视频名 ---
            # current_frame_path 现在是完整路径，例如 "Videos-Frames/my_video_Frames"
            # os.path.basename 获取 "my_video_Frames"
            folder_name = os.path.basename(current_frame_path)
            # replace 去掉后缀，得到纯净的视频名 "my_video"
            video_name = folder_name.replace('_Frames', '')

            print(f"\n[{idx}/{len(frame_folder_paths)}] 处理中...")

            # 调用新的函数名: evaluate_extracted_frames
            result = self.evaluate_extracted_frames(
                frame_folder_path=current_frame_path,
                video_name=video_name
            )
            results.append(result)

            if result['status'] == 'success':
                success_count += 1
                print(f"✓ 评估成功")
                # 打印评估结果摘要（前100字符）
                result_text = result.get('evaluation_result', '')
                preview = result_text[:100] + '...' if len(result_text) > 100 else result_text
                print(f"  结果预览: {preview}")
            else:
                failed_count += 1
                print(f"✗ 评估失败: {result.get('error', '未知错误')}")

        # 保存结果
        output_data = {
            'evaluation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_videos': len(frame_folder_paths),
            'success_count': success_count,
            'failed_count': failed_count,
            'results': results
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        # 打印总结
        print(f"\n{'=' * 60}")
        print(f"评估完成！")
        print(f"总数: {len(frame_folder_paths)} | 成功: {success_count} | 失败: {failed_count}")
        print(f"结果已保存至: {output_file}")
        print(f"{'=' * 60}")

        return results


def main():
    """主函数"""
    # 配置参数
    MODEL_PATH = "./autodl-tmp/Qwen/Qwen3-VL-8B-Instruct"  # 模型路径
    ROOT_FRAME_DIR = "Videos-Frames"  # 视频帧根目录 (改名以体现根目录概念)
    OUTPUT_FILE = "safety_evaluation_results.json"  # 结果输出文件

    # 可选：自定义评估提示词
    CUSTOM_PROMPT = None  # 如果设为 None，则使用默认提示词

    # 创建评估器并执行批量评估
    evaluator = VideoSafetyEvaluator(model_path=MODEL_PATH)
    # 调用 batch_evaluate 时传入新的参数名 root_frame_dir
    results = evaluator.batch_evaluate(
        root_frame_dir=ROOT_FRAME_DIR,
        output_file=OUTPUT_FILE
    )

    # 打印简要结果
    if results:
        print("\n--- 详细结果列表 ---")
        for r in results:
            status_icon = "✓" if r['status'] == 'success' else "✗"
            print(f"{status_icon} {r['video_name']}: {r['status']} ({r['frame_count']} 帧)")


if __name__ == "__main__":
    main()
