#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LTX-2 批量生成视频脚本（基于 T2VSafetyBench）
只加载一次模型，批量生成所有视频
输出文件以 index 命名: 1.mp4, 2.mp4, ...
"""

import json
import os
import sys
import argparse
import logging
import random
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

import torch

# ============================================================
# LTX-2 Pipeline 导入
# ============================================================
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.quantization import QuantizationPolicy


def load_prompts(json_file: str) -> list:
    """从 JSON 文件加载 prompts"""
    with open(json_file, 'r', encoding='utf-8') as f:
        prompts = json.load(f)
    return prompts


def init_logging(rank: int = 0):
    """初始化日志"""
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)],
        )
    else:
        logging.basicConfig(level=logging.ERROR)


def build_video_guider_params(
    cfg_scale: float = 3.0,
    stg_scale: float = 1.0,
    rescale_scale: float = 0.7,
    stg_blocks: list = None,
    skip_step: int = 0,
) -> MultiModalGuiderParams:
    """构建视频引导参数（仅视频，无音频，modality_scale 设为 1.0 禁用）"""
    return MultiModalGuiderParams(
        cfg_scale=cfg_scale,
        stg_scale=stg_scale,
        rescale_scale=rescale_scale,
        modality_scale=1.0,  # 纯视频生成，禁用模态引导
        skip_step=skip_step,
        stg_blocks=stg_blocks or [29],
    )


def build_audio_guider_params() -> MultiModalGuiderParams:
    """构建音频引导参数（纯视频生成时使用默认值即可）"""
    return MultiModalGuiderParams(
        cfg_scale=7.0,
        stg_scale=1.0,
        rescale_scale=0.7,
        modality_scale=1.0,
        skip_step=0,
        stg_blocks=[29],
    )


def batch_generate(
    json_file: str,
    checkpoint_path: str,
    distilled_lora_path: str,
    spatial_upsampler_path: str,
    gemma_root: str,
    lora_paths: list = None,
    quantization: str = None,  # "fp8-cast", "fp8-scaled-mm", or None
    # --- 生成参数 ---
    height: int = 512,
    width: int = 768,
    num_frames: int = 97,       # 帧数，需满足 8k+1 格式（如 97, 193）
    frame_rate: float = 25.0,
    num_inference_steps: int = 40,
    # --- 引导参数 ---
    cfg_scale: float = 3.0,
    stg_scale: float = 1.0,
    rescale_scale: float = 0.7,
    stg_blocks: list = None,
    skip_step: int = 0,
    # --- 筛选参数 ---
    output_dir: str = "./output_videos",
    start_index: int = None,
    end_index: int = None,
    categories: list = None,
    # --- 随机种子 ---
    base_seed: int = -1,
    # --- LoRA 强度 ---
    distilled_lora_strength: float = 0.6,
):
    """
    批量生成视频：只加载一次 Pipeline，遍历所有 prompt 进行生成。

    Parameters
    ----------
    json_file : str
        T2VSafetyBench JSON 文件路径
    checkpoint_path : str
        LTX-2 主模型 checkpoint 路径（.safetensors）
    distilled_lora_path : str
        Distilled LoRA 路径
    spatial_upsampler_path : str
        空间上采样器路径
    gemma_root : str
        Gemma 文本编码器根目录
    lora_paths : list, optional
        额外 LoRA 路径列表
    quantization : str, optional
        量化策略: "fp8-cast", "fp8-scaled-mm", 或 None
    height / width : int
        视频分辨率
    num_frames : int
        帧数（需满足 8k+1，如 97, 129, 161, 193）
    frame_rate : float
        帧率
    num_inference_steps : int
        推理步数（Stage 1）
    cfg_scale / stg_scale / rescale_scale : float
        多模态引导参数
    stg_blocks : list
        STG 作用的 transformer block 列表
    skip_step : int
        每 N 步跳过引导（0 = 不跳过）
    output_dir : str
        输出目录
    start_index / end_index : int, optional
        筛选 index 范围 [start_index, end_index)
    categories : list, optional
        筛选类别列表
    base_seed : int
        基础随机种子（-1 = 随机）
    distilled_lora_strength : float
        Distilled LoRA 强度
    """
    init_logging(rank=0)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # ----------------------------------------------------------------
    # 1. 加载 prompts 并筛选
    # ----------------------------------------------------------------
    prompts = load_prompts(json_file)
    logging.info(f"总共加载了 {len(prompts)} 个 prompts")

    if categories:
        prompts = [p for p in prompts if p.get('category') in categories]
        logging.info(f"筛选出 {len(prompts)} 个指定类别的 prompts: {categories}")

    if start_index is not None or end_index is not None:
        filtered = []
        for p in prompts:
            idx = p['index']
            if start_index is not None and idx < start_index:
                continue
            if end_index is not None and idx >= end_index:
                continue
            filtered.append(p)
        prompts = filtered
        logging.info(f"筛选出 {len(prompts)} 个指定 index 范围的 prompts")

    if len(prompts) == 0:
        logging.warning("没有匹配的 prompt，退出。")
        return

    # ----------------------------------------------------------------
    # 2. 构建量化策略
    # ----------------------------------------------------------------
    quantization_policy = None
    if quantization == "fp8-cast":
        quantization_policy = QuantizationPolicy.fp8_cast()
        logging.info("使用 FP8 Cast 量化策略")
    elif quantization == "fp8-scaled-mm":
        quantization_policy = QuantizationPolicy.fp8_scaled_mm()
        logging.info("使用 FP8 Scaled MM 量化策略（需要 tensorrt_llm）")

    # ----------------------------------------------------------------
    # 3. 构建 Distilled LoRA
    # ----------------------------------------------------------------
    distilled_lora = [
        LoraPathStrengthAndSDOps(
            distilled_lora_path,
            distilled_lora_strength,
            LTXV_LORA_COMFY_RENAMING_MAP,
        ),
    ]

    # ----------------------------------------------------------------
    # 4. 构建引导参数
    # ----------------------------------------------------------------
    video_guider_params = build_video_guider_params(
        cfg_scale=cfg_scale,
        stg_scale=stg_scale,
        rescale_scale=rescale_scale,
        stg_blocks=stg_blocks,
        skip_step=skip_step,
    )
    audio_guider_params = build_audio_guider_params()

    # ----------------------------------------------------------------
    # 5. 关键步骤：只加载一次 Pipeline！
    # ----------------------------------------------------------------
    logging.info("=" * 60)
    logging.info("正在加载 LTX-2 Pipeline（只加载一次）...")
    logging.info("=" * 60)

    logging.info(f"  Checkpoint : {checkpoint_path}")
    logging.info(f"  Gemma Root : {gemma_root}")
    logging.info(f"  Upsampler  : {spatial_upsampler_path}")
    logging.info(f"  Distill LoRA: {distilled_lora_path}")
    logging.info(f"  Resolution : {width}x{height}")
    logging.info(f"  Frames     : {num_frames}")
    logging.info(f"  Steps      : {num_inference_steps}")
    logging.info(f"  Quantization: {quantization or 'None (BF16)'}")

    pipeline = TI2VidTwoStagesPipeline(
        checkpoint_path=checkpoint_path,
        distilled_lora=distilled_lora,
        spatial_upsampler_path=spatial_upsampler_path,
        gemma_root=gemma_root,
        loras=lora_paths or [],
        quantization=quantization_policy,
    )

    logging.info("✓ Pipeline 加载完成！")
    logging.info("=" * 60)

    # ----------------------------------------------------------------
    # 6. 批量生成视频
    # ----------------------------------------------------------------
    success_count = 0
    fail_count = 0

    for i, prompt_data in enumerate(prompts):
        prompt_idx = prompt_data['index']
        prompt_text = prompt_data['prompt']
        category = prompt_data.get('category', 'N/A')

        logging.info(f"{'=' * 60}")
        logging.info(f"[{i + 1}/{len(prompts)}] 生成中...")
        logging.info(f"  Index   : {prompt_idx}")
        logging.info(f"  Category: {category}")
        logging.info(f"  Prompt  : {prompt_text[:100]}{'...' if len(prompt_text) > 100 else ''}")
        logging.info(f"{'=' * 60}")

        # 设置随机种子
        seed = base_seed if base_seed >= 0 else random.randint(0, 2**32 - 1)

        # 输出文件: {index}.mp4
        output_path = os.path.join(output_dir, f"{prompt_idx}.mp4")

        # 如果已存在，跳过
        if os.path.exists(output_path):
            logging.info(f"  ⏭ 已存在，跳过: {output_path}")
            success_count += 1
            continue

        try:
            pipeline(
                prompt=prompt_text,
                output_path=output_path,
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                num_inference_steps=num_inference_steps,
                video_guider_params=video_guider_params,
                audio_guider_params=audio_guider_params,
            )

            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                logging.info(f"  ✓ 成功: {output_path} ({file_size_mb:.2f} MB)")
                success_count += 1
            else:
                logging.warning(f"  ✗ 输出文件异常: {output_path}")
                fail_count += 1

        except Exception as e:
            logging.error(f"  ✗ 生成失败 (Index {prompt_idx}): {str(e)}")
            import traceback
            traceback.print_exc()
            fail_count += 1
            # 删除可能生成的空文件
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except OSError:
                    pass

        # 清理显存
        torch.cuda.empty_cache()

    # ----------------------------------------------------------------
    # 7. 汇总
    # ----------------------------------------------------------------
    logging.info(f"{'=' * 60}")
    logging.info("批量生成完成！")
    logging.info(f"  总计: {len(prompts)} 个")
    logging.info(f"  成功: {success_count} 个")
    logging.info(f"  失败: {fail_count} 个")
    logging.info(f"  输出目录: {output_dir}")
    logging.info(f"{'=' * 60}")


# ==============================================================
# CLI 入口
# ==============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LTX-2 批量生成视频（基于 T2VSafetyBench）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本用法：全量生成
  python ltx2_batch_generate.py \\
      --json_file T2VSafetyBench_sampled_3class.json \\
      --checkpoint_path ./models/ltx-2.3-22b-dev.safetensors \\
      --distilled_lora_path ./models/ltx-2.3-22b-distilled-lora-384-1.1.safetensors \\
      --spatial_upsampler_path ./models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \\
      --gemma_root ./models/gemma-3-12b-it-qat-q4_0-unquantized \\
      --output_dir ./output_videos

  # 只生成 Violence 类别，index 31~45
  python ltx2_batch_generate.py \\
      --json_file T2VSafetyBench_sampled_3class.json \\
      --checkpoint_path ./models/ltx-2.3-22b-dev.safetensors \\
      --distilled_lora_path ./models/ltx-2.3-22b-distilled-lora-384-1.1.safetensors \\
      --spatial_upsampler_path ./models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \\
      --gemma_root ./models/gemma-3-12b-it-qat-q4_0-unquantized \\
      --categories Violence \\
      --start_index 31 --end_index 45 \\
      --output_dir ./output_violence

  # 使用 FP8 量化降低显存
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python ltx2_batch_generate.py \\
      --json_file T2VSafetyBench_sampled_3class.json \\
      --checkpoint_path ./models/ltx-2.3-22b-dev.safetensors \\
      --distilled_lora_path ./models/ltx-2.3-22b-distilled-lora-384-1.1.safetensors \\
      --spatial_upsampler_path ./models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \\
      --gemma_root ./models/gemma-3-12b-it-qat-q4_0-unquantized \\
      --quantization fp8-cast \\
      --output_dir ./output_videos

  # 更少步数、更小分辨率（快速测试）
  python ltx2_batch_generate.py \\
      --json_file T2VSafetyBench_sampled_3class.json \\
      --checkpoint_path ./models/ltx-2.3-22b-dev.safetensors \\
      --distilled_lora_path ./models/ltx-2.3-22b-distilled-lora-384-1.1.safetensors \\
      --spatial_upsampler_path ./models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \\
      --gemma_root ./models/gemma-3-12b-it-qat-q4_0-unquantized \\
      --width 512 --height 384 \\
      --num_frames 97 \\
      --num_inference_steps 20 \\
      --output_dir ./output_test
        """,
    )

    # ===== 必选参数 =====
    parser.add_argument(
        "--json_file", type=str, required=True,
        help="T2VSafetyBench JSON 文件路径"
    )
    parser.add_argument(
        "--checkpoint_path", type=str, required=True,
        help="LTX-2 模型 checkpoint 路径 (.safetensors)"
    )
    parser.add_argument(
        "--distilled_lora_path", type=str, required=True,
        help="Distilled LoRA 路径 (.safetensors)"
    )
    parser.add_argument(
        "--spatial_upsampler_path", type=str, required=True,
        help="空间上采样器路径 (.safetensors)"
    )
    parser.add_argument(
        "--gemma_root", type=str, required=True,
        help="Gemma 文本编码器根目录"
    )

    # ===== 生成参数 =====
    parser.add_argument("--width", type=int, default=768, help="视频宽度 (默认: 768)")
    parser.add_argument("--height", type=int, default=512, help="视频高度 (默认: 512)")
    parser.add_argument(
        "--num_frames", type=int, default=97,
        help="帧数，需满足 8k+1 格式 (如 97, 193)，默认: 97"
    )
    parser.add_argument("--frame_rate", type=float, default=25.0, help="帧率 (默认: 25.0)")
    parser.add_argument(
        "--num_inference_steps", type=int, default=40,
        help="推理步数，越高质量越好但越慢 (默认: 40)"
    )

    # ===== 引导参数 =====
    parser.add_argument("--cfg_scale", type=float, default=3.0, help="CFG 引导强度 (默认: 3.0)")
    parser.add_argument("--stg_scale", type=float, default=1.0, help="STG 引导强度 (默认: 1.0)")
    parser.add_argument("--rescale_scale", type=float, default=0.7, help="重缩放比例 (默认: 0.7)")
    parser.add_argument("--skip_step", type=int, default=0, help="每 N 步跳过引导，0=不跳过 (默认: 0)")
    parser.add_argument(
        "--stg_blocks", type=int, nargs='+', default=[29],
        help="STG 作用的 transformer block 列表 (默认: [29])"
    )

    # ===== 量化 =====
    parser.add_argument(
        "--quantization", type=str, default=None,
        choices=["fp8-cast", "fp8-scaled-mm", None],
        help="量化策略: fp8-cast / fp8-scaled-mm / None (默认: None)"
    )

    # ===== LoRA =====
    parser.add_argument(
        "--lora_paths", type=str, nargs='*', default=None,
        help="额外 LoRA 文件路径列表"
    )
    parser.add_argument(
        "--distilled_lora_strength", type=float, default=0.6,
        help="Distilled LoRA 强度 (默认: 0.6)"
    )

    # ===== 筛选 =====
    parser.add_argument("--output_dir", type=str, default="./output_videos", help="输出目录")
    parser.add_argument("--start_index", type=int, default=None, help="起始 index（包含）")
    parser.add_argument("--end_index", type=int, default=None, help="结束 index（不包含）")
    parser.add_argument(
        "--categories", type=str, nargs='+', default=None,
        help="筛选类别 (如: Pornography Violence Gore)"
    )

    # ===== 种子 =====
    parser.add_argument(
        "--base_seed", type=int, default=-1,
        help="基础随机种子，-1=随机 (默认: -1)"
    )

    args = parser.parse_args()

    batch_generate(
        json_file=args.json_file,
        checkpoint_path=args.checkpoint_path,
        distilled_lora_path=args.distilled_lora_path,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        lora_paths=args.lora_paths,
        quantization=args.quantization,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.cfg_scale,
        stg_scale=args.stg_scale,
        rescale_scale=args.rescale_scale,
        stg_blocks=args.stg_blocks,
        skip_step=args.skip_step,
        output_dir=args.output_dir,
        start_index=args.start_index,
        end_index=args.end_index,
        categories=args.categories,
        base_seed=args.base_seed,
        distilled_lora_strength=args.distilled_lora_strength,
    )
