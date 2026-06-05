#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Helios批量视频生成脚本
用于批量处理T2VSafetyBench数据集
参考: infer_helios.py 和 wan2.2_video_generate.py
"""

import importlib
import os
import json
import sys
import argparse
import logging
import random
import time
from datetime import datetime
from tqdm import tqdm

# ========== 新增：禁用 flash-attn 自动下载 + 修复 OMP 环境变量 ==========
# 强制设置 OMP_NUM_THREADS
os.environ["OMP_NUM_THREADS"] = "8"  # 根据CPU核心数调整
# 禁用 kernels 自动下载
os.environ["KERNELS_DISABLE_AUTOINSTALL"] = "1"
# 禁用 flash-attn 相关优化
os.environ["HELIOS_DISABLE_FLASH_ATTN"] = "1"
# =====================================================================

# 设置环境变量
os.environ["HF_ENABLE_PARALLEL_LOADING"] = "yes"
os.environ["HF_PARALLEL_LOADING_WORKERS"] = "8"

if importlib.util.find_spec("torch_npu") is not None:
    import torch_npu
else:
    torch_npu = None

import torch
import torch.distributed as dist
from PIL import Image

from helios.diffusers_version.pipeline_helios_diffusers import HeliosPipeline
from helios.diffusers_version.scheduling_helios_diffusers import HeliosScheduler
from helios.diffusers_version.transformer_helios_diffusers import HeliosTransformer3DModel
from helios.modules.helios_kernels import (
    replace_all_norms_with_flash_norms,
    replace_rmsnorm_with_fp32,
    replace_rope_with_flash_rope,
)
from helios.utils.utils_base import load_extra_components

from diffusers.models import AutoencoderKLWan
from diffusers.utils import export_to_video, load_image


def load_prompts(json_file):
    """从JSON文件加载prompts"""
    with open(json_file, 'r', encoding='utf-8') as f:
        prompts = json.load(f)
    return prompts


def init_logging(rank):
    """初始化日志"""
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Helios批量视频生成脚本")

    # === 数据相关参数 ===
    parser.add_argument(
        "--json_file",
        type=str,
        required=True,
        help="T2VSafetyBench JSON文件路径"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="./output_helios",
        help="输出视频文件夹"
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=None,
        help="起始index（包含）"
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=None,
        help="结束index（不包含）"
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs='+',
        default=None,
        help="要处理的类别列表，如 Pornography Violence Gore"
    )

    # === 模型路径参数 ===
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="/root/autodl-tmp/Helios-Base",
        help="基础模型路径"
    )
    parser.add_argument(
        "--transformer_path",
        type=str,
        default="/root/autodl-tmp/Helios-Base",
        help="Transformer模型路径"
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="LoRA权重路径"
    )
    parser.add_argument(
        "--partial_path",
        type=str,
        default=None,
        help="Partial权重路径"
    )

    # === 生成参数 ===
    parser.add_argument(
        "--weight_dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="模型权重数据类型"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=384,
        help="视频高度"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="视频宽度"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=99,
        help="视频帧数"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="视频帧率"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="推理步数"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=5.0,
        help="引导比例"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
        help="负面提示词"
    )

    # === 技术优化参数 ===
    parser.add_argument(
        "--enable_compile",
        action="store_true",
        help="启用模型编译"
    )
    parser.add_argument(
        "--enable_low_vram_mode",
        action="store_true",
        help="启用低显存模式"
    )
    parser.add_argument(
        "--group_offloading_type",
        type=str,
        choices=["leaf_level", "block_level"],
        default="leaf_level",
        help="组卸载类型"
    )

    # === 分布式参数 ===
    parser.add_argument(
        "--enable_parallelism",
        action="store_true",
        help="启用并行处理"
    )
    parser.add_argument(
        "--cp_backend",
        type=str,
        choices=["ring", "ulysses", "unified", "ulysses_anything"],
        default="ulysses",
        help="上下文并行后端"
    )

    return parser.parse_args()


def setup_device_and_distributed(args):
    """设置设备和分布式环境"""
    if dist.is_available() and "RANK" in os.environ:
        if args.cp_backend == "ulysses_anything":
            dist.init_process_group(backend="cpu:gloo,cuda:nccl")
        else:
            dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        device = torch.device("cuda", rank % torch.cuda.device_count())
        world_size = dist.get_world_size()
        torch.cuda.set_device(device)
    else:
        rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        world_size = 1

    return rank, device, world_size


def load_helios_model(args, device):
    """加载Helios模型"""
    # 设置数据类型
    if args.weight_dtype == "fp32":
        weight_dtype = torch.float32
    elif args.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.bfloat16

    logging.info(f"加载Transformer模型: {args.transformer_path}")
    transformer = HeliosTransformer3DModel.from_pretrained(
        args.transformer_path,
        subfolder="transformer",
        torch_dtype=weight_dtype,
    )

    # 应用优化
    if not args.enable_compile:
        transformer = replace_rmsnorm_with_fp32(transformer)
        # 注释/删除 flash norm/rope 替换（避免依赖 flash-attn）
        # transformer = replace_all_norms_with_flash_norms(transformer)
        # replace_rope_with_flash_rope()

    # 设置注意力后端
    # cuda_major = torch.cuda.get_device_capability()[0]
    # if cuda_major >= 9:
    #     try:
    #         transformer.set_attention_backend("_flash_3_hub")
    #     except Exception:
    #         transformer.set_attention_backend("flash_hub")
    # else:
    #     transformer.set_attention_backend("flash_hub")
    # 强制使用flash_hub，跳过flash-attn3的HuggingFace检查
    transformer.set_attention_backend("flash_hub")

    # 加载VAE和Scheduler
    logging.info(f"加载VAE和Scheduler: {args.base_model_path}")
    vae = AutoencoderKLWan.from_pretrained(
        args.base_model_path,
        subfolder="vae",
        torch_dtype=torch.float32,
    )

    scheduler = HeliosScheduler.from_pretrained(
        args.base_model_path,
        subfolder="scheduler",
    )

    # 创建管道
    logging.info("创建HeliosPipeline")
    pipe = HeliosPipeline.from_pretrained(
        args.base_model_path,
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
        torch_dtype=weight_dtype,
    )

    # 加载LoRA权重
    if args.lora_path is not None:
        logging.info(f"加载LoRA权重: {args.lora_path}")
        pipe.load_lora_weights(args.lora_path, adapter_name="default")
        pipe.set_adapters(["default"], adapter_weights=[1.0])

        if args.partial_path is not None:
            from argparse import Namespace
            args.training_config = Namespace()
            args.training_config.is_enable_stage1 = True
            args.training_config.restrict_self_attn = True
            args.training_config.is_amplify_history = True
            args.training_config.is_use_gan = True
            load_extra_components(args, transformer, args.partial_path)

    # 模型编译
    if args.enable_compile:
        logging.info("启用模型编译")
        torch.backends.cudnn.benchmark = True
        pipe.text_encoder.compile(mode="max-autotune-no-cudagraphs", dynamic=False)
        pipe.vae.compile(mode="max-autotune-no-cudagraphs", dynamic=False)
        pipe.transformer.compile(mode="max-autotune-no-cudagraphs", dynamic=False)

    # 低显存模式
    if args.enable_low_vram_mode:
        logging.info("启用低显存模式")
        pipe.enable_group_offload(
            onload_device=torch.device("cuda"),
            offload_device=torch.device("cpu"),
            offload_type=args.group_offloading_type,
            num_blocks_per_group=None if args.group_offloading_type == "leaf_level" else 4,
            use_stream=True,
            record_stream=True,
        )
    else:
        pipe = pipe.to(device)

    return pipe, weight_dtype


def filter_prompts(prompts, start_index, end_index, categories):
    """筛选prompts"""
    filtered_prompts = []

    for p in prompts:
        idx = p['index']

        # 索引筛选
        if start_index is not None and idx < start_index:
            continue
        if end_index is not None and idx >= end_index:
            continue

        # 类别筛选
        if categories and p.get('category') not in categories:
            continue

        filtered_prompts.append(p)

    return filtered_prompts


def batch_generate_helios(args):
    """批量生成视频主函数"""
    # 设置设备和分布式
    rank, device, world_size = setup_device_and_distributed(args)
    init_logging(rank)

    # 创建输出目录
    if rank == 0:
        os.makedirs(args.output_folder, exist_ok=True)

    # 加载prompts
    if rank == 0:
        logging.info(f"加载JSON文件: {args.json_file}")
    prompts = load_prompts(args.json_file)

    if rank == 0:
        logging.info(f"总共加载了 {len(prompts)} 个prompts")

    # 筛选prompts
    filtered_prompts = filter_prompts(
        prompts, args.start_index, args.end_index, args.categories
    )

    if rank == 0:
        logging.info(f"筛选后剩余 {len(filtered_prompts)} 个prompts")

    # 分布式数据划分
    if not args.enable_parallelism and world_size > 1:
        my_prompts = filtered_prompts[rank::world_size]
    else:
        my_prompts = filtered_prompts

    if rank == 0:
        logging.info(f"当前进程处理 {len(my_prompts)} 个prompts")

    # 只加载一次模型
    if rank == 0:
        logging.info("=" * 60)
        logging.info("正在加载Helios模型（只加载一次）...")
        logging.info("=" * 60)

    pipe, weight_dtype = load_helios_model(args, device)

    if rank == 0:
        logging.info("✓ Helios模型加载完成！")
        logging.info("=" * 60)

    # 批量生成视频
    for i, prompt_data in enumerate(my_prompts):
        idx = prompt_data['index']
        prompt_text = prompt_data['prompt']
        category = prompt_data.get('category', 'N/A')

        if rank == 0:
            logging.info(f"{'=' * 60}")
            logging.info(f"正在生成第 {i + 1}/{len(my_prompts)} 个视频")
            logging.info(f"Index: {idx}")
            logging.info(f"Category: {category}")
            logging.info(f"Prompt: {prompt_text[:100]}...")
            logging.info(f"{'=' * 60}")

        # 检查文件是否已存在
        output_path = os.path.join(args.output_folder, f"{idx}.mp4")
        if os.path.exists(output_path):
            if rank == 0:
                logging.info(f"文件已存在，跳过: {output_path}")
            continue

        try:
            # 生成视频
            with torch.no_grad():
                output = pipe(
                    prompt=prompt_text,
                    negative_prompt=args.negative_prompt,
                    height=args.height,
                    width=args.width,
                    num_frames=args.num_frames,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    generator=torch.Generator(device="cuda").manual_seed(args.seed + idx),
                    # stage 1
                    history_sizes=[16, 2, 1],
                    num_latent_frames_per_chunk=9,  # 默认值
                    keep_first_frame=True,
                    # stage 2
                    is_enable_stage2=False,  # 默认值
                    pyramid_num_inference_steps_list=[20, 20, 20],  # 默认值
                    # stage 3
                    is_skip_first_chunk=False,  # 默认值
                    is_amplify_first_chunk=False,  # 默认值
                    # cfg zero
                    use_zero_init=False,  # 默认值
                    zero_steps=1,  # 默认值
                ).frames[0]

            # 保存视频
            if not args.enable_parallelism or rank == 0:
                export_to_video(output, output_path, fps=args.fps)
                if rank == 0:
                    logging.info(f"✓ 成功保存视频: {output_path}")

            # 清理显存
            del output
            torch.cuda.synchronize()

        except Exception as e:
            if rank == 0:
                logging.error(f"✗ 生成失败 (Index {idx}): {str(e)}")
                import traceback
                traceback.print_exc()
            continue

    # 输出显存使用情况
    if torch.cuda.is_available():
        max_memory_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
        if rank == 0:
            logging.info(f"最大显存使用: {max_memory_gb:.3f} GB")

    if rank == 0:
        logging.info(f"{'=' * 60}")
        logging.info("批量生成完成！")
        logging.info(f"输出目录: {args.output_folder}")
        logging.info(f"{'=' * 60}")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    batch_generate_helios(args)