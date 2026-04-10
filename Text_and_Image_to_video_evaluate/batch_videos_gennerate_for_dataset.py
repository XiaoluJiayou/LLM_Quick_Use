#!/usr/bin/env python3
"""
Qwen3-VL 本地视频生成脚本
功能：读取JSON文件，基于本地权重生成视频，保存为 {index}.mp4
作者：AI Assistant
"""

import json
import os
import sys
import torch
import numpy as np
import cv2
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import logging
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# 默认配置
# ============================================================================

DEFAULT_CONFIG = {
    # 模型配置
    'model_path': '/path/to/qwen3-vl-weights',  # 修改为你的模型路径
    'device': 'cuda',  # cuda 或 cpu
    'dtype': 'float16',  # float16, float32, bfloat16

    # 生成参数
    'num_frames': 16,  # 视频帧数
    'fps': 8,  # 帧率
    'height': 720,  # 视频高度
    'width': 1280,  # 视频宽度
    'guidance_scale': 7.5,  # 引导系数
    'num_inference_steps': 50,  # 推理步数
    'seed': None,  # 随机种子

    # 数据配置
    'json_file': 'T2VSafetyBench_sampled.json',
    'output_dir': './generated_videos',
    'start_index': 0,
    'end_index': None,

    # 性能优化
    'use_xformers': False,
    'enable_vae_slicing': True,
}


# ============================================================================
# 视频生成器类
# ============================================================================

class LocalVideoGenerator:
    """基于本地权重的视频生成器"""

    def __init__(self, config: Dict):
        """
        初始化视频生成器

        Args:
            config: 配置字典
        """
        self.config = config
        self.model_path = config['model_path']
        self.device = config['device']
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 设置数据类型
        dtype_str = config.get('dtype', 'float16')
        self.dtype = torch.float16 if dtype_str == "float16" else (
            torch.bfloat16 if dtype_str == "bfloat16" else torch.float32
        )

        # 模型组件
        self.model = None
        self.processor = None

        # 检查设备
        if self.device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA不可用，切换到CPU模式")
            self.device = 'cpu'

        logger.info(f"使用设备: {self.device}")
        logger.info(f"数据类型: {dtype_str}")

    def load_model(self):
        """加载本地模型权重"""
        logger.info(f"正在从 {self.model_path} 加载模型...")

        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"模型路径不存在: {self.model_path}")

        try:
            # 方式1: 使用transformers库加载（推荐）
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info("使用 transformers 加载模型...")

            # 加载tokenizer
            self.processor = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            logger.info("Tokenizer 加载成功")

            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=self.dtype,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )

            if self.device == "cpu":
                self.model = self.model.to(self.device)

            logger.info("模型加载成功!")

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            logger.error("\n请确保:")
            logger.error("1. 模型路径正确")
            logger.error("2. 已安装 transformers 库")
            logger.error("3. 模型文件完整")
            raise

    def load_json_file(self, json_file_path: str) -> List[Dict]:
        """
        读取JSON文件

        Args:
            json_file_path: JSON文件路径

        Returns:
            包含所有条目的列表
        """
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"成功读取JSON文件，共 {len(data)} 条数据")
            return data
        except Exception as e:
            logger.error(f"读取JSON文件失败: {e}")
            raise

    @torch.no_grad()
    def generate_video(
            self,
            prompt: str,
            num_frames: Optional[int] = None,
            fps: Optional[int] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            guidance_scale: Optional[float] = None,
            num_inference_steps: Optional[int] = None,
            seed: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """
        生成视频

        Args:
            prompt: 文本提示词
            num_frames: 帧数
            fps: 帧率
            height: 视频高度
            width: 视频宽度
            guidance_scale: 引导系数
            num_inference_steps: 推理步数
            seed: 随机种子

        Returns:
            视频帧数组 [T, H, W, C]
        """
        # 使用配置中的默认值
        num_frames = num_frames or self.config['num_frames']
        fps = fps or self.config['fps']
        height = height or self.config['height']
        width = width or self.config['width']
        guidance_scale = guidance_scale or self.config['guidance_scale']
        num_inference_steps = num_inference_steps or self.config['num_inference_steps']
        seed = seed if seed is not None else self.config['seed']

        try:
            logger.info(f"生成视频: {prompt[:60]}...")

            # 设置随机种子
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

            # ======== 方式1: 如果模型有 generate_video 方法 ========
            if hasattr(self.model, 'generate_video'):
                video_frames = self.model.generate_video(
                    prompt=prompt,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps
                )

            # ======== 方式2: 如果模型是扩散模型 ========
            elif hasattr(self.model, 'generate'):
                # 编码文本
                inputs = self.processor(
                    prompt,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)

                text_embeddings = self.model.get_text_features(**inputs)

                # 生成
                video_frames = self.model.generate(
                    text_embeddings=text_embeddings,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps
                )

            # ======== 方式3: 通用扩散生成流程 ========
            else:
                # 初始化潜在空间噪声
                latents = torch.randn(
                    (num_frames, 3, height // 8, width // 8),
                    device=self.device,
                    dtype=self.dtype
                )

                # 编码文本
                inputs = self.processor(
                    prompt,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)

                text_embeddings = self.model.encode_prompt(inputs)

                # 扩散去噪过程
                for step in tqdm(range(num_inference_steps), desc="去噪步骤"):
                    # 预测噪声
                    noise_pred = self.model(
                        latents,
                        timestep=step,
                        encoder_hidden_states=text_embeddings
                    )

                    # 更新latents (简化的DDPM步骤)
                    alpha = 1.0 - step / 100
                    latents = latents - alpha * noise_pred

                # 解码为视频帧
                video_frames = self.model.decode(latents)

            # 转换为numpy数组
            if isinstance(video_frames, torch.Tensor):
                video_frames = video_frames.cpu().numpy()

            # 确保格式正确 [T, H, W, C], 值范围 [0, 255]
            if video_frames.max() <= 1.0:
                video_frames = (video_frames * 255).astype(np.uint8)
            else:
                video_frames = video_frames.astype(np.uint8)

            logger.info(f"视频生成成功，形状: {video_frames.shape}")
            return video_frames

        except Exception as e:
            logger.error(f"生成视频失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def save_video(
            self,
            video_frames: np.ndarray,
            index: int,
            fps: Optional[int] = None
    ) -> str:
        """
        保存视频文件

        Args:
            video_frames: 视频帧数组 [T, H, W, C]
            index: 索引值
            fps: 帧率

        Returns:
            保存的文件路径
        """
        fps = fps or self.config['fps']
        filename = f"{index}.mp4"
        filepath = self.output_dir / filename

        # 获取视频尺寸
        num_frames, height, width, channels = video_frames.shape

        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            str(filepath),
            fourcc,
            fps,
            (width, height)
        )

        # 写入每一帧
        for frame in video_frames:
            # RGB转BGR
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        out.release()
        logger.info(f"视频已保存: {filepath}")

        return str(filepath)

    def process_json_file(
            self,
            json_file_path: Optional[str] = None,
            start_index: Optional[int] = None,
            end_index: Optional[int] = None
    ):
        """
        批量处理JSON文件中的所有条目

        Args:
            json_file_path: JSON文件路径
            start_index: 开始处理的索引
            end_index: 结束处理的索引（不包含）
        """
        json_file_path = json_file_path or self.config['json_file']
        start_index = start_index if start_index is not None else self.config['start_index']
        end_index = end_index if end_index is not None else self.config['end_index']

        # 读取数据
        data = self.load_json_file(json_file_path)

        if end_index is None:
            end_index = len(data)

        total = min(end_index, len(data)) - start_index
        logger.info(f"\n{'=' * 70}")
        logger.info(f"开始处理 {total} 条数据")
        logger.info(f"{'=' * 70}")

        success_count = 0
        fail_count = 0
        skip_count = 0

        for i, item in enumerate(data[start_index:end_index], start=start_index):
            index = item.get("index")
            prompt = item.get("prompt", "")
            category = item.get("category", "")

            logger.info(f"\n{'=' * 70}")
            logger.info(f"处理进度: {i - start_index + 1}/{total}")
            logger.info(f"Index: {index}, Category: {category}")
            logger.info(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
            logger.info(f"{'=' * 70}")

            # 检查是否已存在
            output_file = self.output_dir / f"{index}.mp4"
            if output_file.exists():
                logger.info(f"文件已存在，跳过: {output_file}")
                skip_count += 1
                continue

            # 生成视频
            video_frames = self.generate_video(prompt)

            if video_frames is not None:
                self.save_video(video_frames, index)
                success_count += 1
                logger.info(f"成功: {success_count}, 失败: {fail_count}, 跳过: {skip_count}")
            else:
                fail_count += 1
                # 记录失败的条目
                failed_log = self.output_dir / "failed_tasks.txt"
                with open(failed_log, 'a', encoding='utf-8') as f:
                    f.write(f"Index: {index}, Prompt: {prompt}, Category: {category}\n")
                logger.error(f"生成失败")

        # 打印总结
        logger.info(f"\n{'=' * 70}")
        logger.info(f"处理完成!")
        logger.info(f"成功: {success_count}, 失败: {fail_count}, 跳过: {skip_count}")
        logger.info(f"{'=' * 70}")

        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================================
# 辅助函数
# ============================================================================

def check_environment():
    """检查运行环境"""
    logger.info("\n" + "=" * 70)
    logger.info("环境检查")
    logger.info("=" * 70)

    issues = []

    # 检查Python版本
    version = sys.version_info
    logger.info(f"Python版本: {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        issues.append("Python版本需要 >= 3.8")

    # 检查PyTorch
    try:
        import torch
        logger.info(f"PyTorch版本: {torch.__version__}")

        if torch.cuda.is_available():
            logger.info(f"CUDA版本: {torch.version.cuda}")
            logger.info(f"GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            logger.warning("CUDA不可用，将使用CPU模式")
    except ImportError:
        issues.append("未安装 PyTorch")

    # 检查其他依赖
    required_packages = ['numpy', 'cv2', 'transformers']
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"已安装: {package}")
        except ImportError:
            issues.append(f"未安装 {package}")

    if issues:
        logger.error("\n环境问题:")
        for issue in issues:
            logger.error(f"  - {issue}")
        logger.error("\n请运行: pip install torch torchvision transformers numpy opencv-python tqdm")
        return False

    logger.info("\n环境检查通过")
    return True


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='Qwen3-VL 本地视频生成脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # 模型参数
    parser.add_argument('--model_path', type=str, help='模型权重路径')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], help='运行设备')
    parser.add_argument('--dtype', type=str, choices=['float16', 'float32', 'bfloat16'], help='数据类型')

    # 数据参数
    parser.add_argument('--json_file', type=str, help='JSON文件路径')
    parser.add_argument('--output_dir', type=str, help='输出目录')
    parser.add_argument('--start_index', type=int, help='开始索引')
    parser.add_argument('--end_index', type=int, help='结束索引')

    # 生成参数
    parser.add_argument('--num_frames', type=int, help='视频帧数')
    parser.add_argument('--fps', type=int, help='帧率')
    parser.add_argument('--height', type=int, help='视频高度')
    parser.add_argument('--width', type=int, help='视频宽度')
    parser.add_argument('--guidance_scale', type=float, help='引导系数')
    parser.add_argument('--num_inference_steps', type=int, help='推理步数')
    parser.add_argument('--seed', type=int, help='随机种子')

    # 其他参数
    parser.add_argument('--check_env', action='store_true', help='只检查环境，不运行')

    args = parser.parse_args()

    return args


def merge_config_with_args(config: Dict, args: argparse.Namespace) -> Dict:
    """合并默认配置和命令行参数"""
    # 只更新非None的参数
    if args.model_path:
        config['model_path'] = args.model_path
    if args.device:
        config['device'] = args.device
    if args.dtype:
        config['dtype'] = args.dtype
    if args.json_file:
        config['json_file'] = args.json_file
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.start_index is not None:
        config['start_index'] = args.start_index
    if args.end_index is not None:
        config['end_index'] = args.end_index
    if args.num_frames:
        config['num_frames'] = args.num_frames
    if args.fps:
        config['fps'] = args.fps
    if args.height:
        config['height'] = args.height
    if args.width:
        config['width'] = args.width
    if args.guidance_scale:
        config['guidance_scale'] = args.guidance_scale
    if args.num_inference_steps:
        config['num_inference_steps'] = args.num_inference_steps
    if args.seed is not None:
        config['seed'] = args.seed

    return config


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()

    # 检查环境
    if args.check_env:
        check_environment()
        return

    # 合并配置
    config = DEFAULT_CONFIG.copy()
    config = merge_config_with_args(config, args)

    # 打印配置
    logger.info("\n" + "=" * 70)
    logger.info("运行配置:")
    logger.info("=" * 70)
    logger.info(f"模型路径: {config['model_path']}")
    logger.info(f"设备: {config['device']}")
    logger.info(f"数据类型: {config['dtype']}")
    logger.info(f"JSON文件: {config['json_file']}")
    logger.info(f"输出目录: {config['output_dir']}")
    logger.info(f"处理范围: {config['start_index']} - {config['end_index'] or 'end'}")
    logger.info(f"视频参数: {config['num_frames']}帧, {config['fps']}fps, "
                f"{config['height']}x{config['width']}")
    logger.info(f"推理参数: guidance_scale={config['guidance_scale']}, "
                f"steps={config['num_inference_steps']}")
    logger.info("=" * 70 + "\n")

    # 检查环境
    if not check_environment():
        sys.exit(1)

    # 检查模型路径
    if not Path(config['model_path']).exists():
        logger.error(f"\n模型路径不存在: {config['model_path']}")
        logger.error("请使用 --model_path 参数指定正确的模型路径")
        logger.error("\n示例: python script.py --model_path /path/to/qwen3-vl")
        sys.exit(1)

    # 检查JSON文件
    if not Path(config['json_file']).exists():
        logger.error(f"\nJSON文件不存在: {config['json_file']}")
        logger.error("请使用 --json_file 参数指定正确的JSON文件路径")
        sys.exit(1)

    try:
        # 创建生成器实例
        generator = LocalVideoGenerator(config)

        # 加载模型
        generator.load_model()

        # 处理JSON文件
        generator.process_json_file()

    except KeyboardInterrupt:
        logger.info("\n用户中断，程序退出")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n程序运行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
