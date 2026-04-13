#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优化版批量生成视频脚本
只加载一次模型，批量生成所有视频
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
import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from wan.utils.utils import save_video

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

def batch_generate_optimized(
    json_file,
    ckpt_dir,
    task='t2v-A14B',
    size='832*480',
    output_dir='./output_videos',
    start_index=None,
    end_index=None,
    categories=None,
    offload_model=False,
    convert_model_dtype=True,
    frame_num=None,
    sample_steps=None,
    sample_shift=None,
    sample_guide_scale=None,
    base_seed=-1
):
    """
    优化版批量生成：只加载一次模型
    """
    # 初始化分布式环境
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    
    init_logging(rank)
    
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size
        )
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载prompts
    prompts = load_prompts(json_file)
    logging.info(f"总共加载了 {len(prompts)} 个prompts")
    
    # 筛选prompts
    if categories:
        prompts = [p for p in prompts if p.get('category') in categories]
        logging.info(f"筛选出 {len(prompts)} 个指定类别的prompts")
    
    if start_index is not None or end_index is not None:
        filtered_prompts = []
        for p in prompts:
            idx = p['index']
            if start_index is not None and idx < start_index:
                continue
            if end_index is not None and idx >= end_index:
                continue
            filtered_prompts.append(p)
        prompts = filtered_prompts
        logging.info(f"筛选出 {len(prompts)} 个指定index范围的prompts")
    
    # 获取模型配置
    cfg = WAN_CONFIGS[task]
    
    # 设置默认值
    if frame_num is None:
        frame_num = cfg.frame_num
    if sample_steps is None:
        sample_steps = cfg.sample_steps
    if sample_shift is None:
        sample_shift = cfg.sample_shift
    if sample_guide_scale is None:
        sample_guide_scale = cfg.sample_guide_scale
    
    # ========================================
    # 关键步骤：只加载一次模型！
    # ========================================
    logging.info("="*60)
    logging.info("正在加载模型（只加载一次）...")
    logging.info("="*60)
    
    if "t2v" in task:
        pipeline = wan.WanT2V(
            config=cfg,
            checkpoint_dir=ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=False,
            convert_model_dtype=convert_model_dtype,
        )
    elif "i2v" in task:
        pipeline = wan.WanI2V(
            config=cfg,
            checkpoint_dir=ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=False,
            convert_model_dtype=convert_model_dtype,
        )
    else:
        raise ValueError(f"Unsupported task: {task}")
    
    logging.info("✓ 模型加载完成！")
    logging.info("="*60)
    
    # ========================================
    # 批量生成视频
    # ========================================
    for i, prompt_data in enumerate(prompts):
        logging.info(f"{'='*60}")
        logging.info(f"正在生成第 {i+1}/{len(prompts)} 个视频")
        logging.info(f"Index: {prompt_data['index']}")
        logging.info(f"Category: {prompt_data.get('category', 'N/A')}")
        logging.info(f"Prompt: {prompt_data['prompt'][:80]}...")
        logging.info(f"{'='*60}")
        
        try:
            # 设置随机种子
            seed = base_seed if base_seed >= 0 else random.randint(0, sys.maxsize)
            
            # 生成视频（使用已加载的pipeline）
            video = pipeline.generate(
                prompt_data['prompt'],
                size=SIZE_CONFIGS[size],
                frame_num=frame_num,
                shift=sample_shift,
                sample_solver='unipc',
                sampling_steps=sample_steps,
                guide_scale=sample_guide_scale,
                seed=seed,
                offload_model=offload_model
            )
            
            # 保存视频
            save_file = os.path.join(output_dir, f"{prompt_data['index']}.mp4")
            logging.info(f"保存视频到: {save_file}")
            
            save_video(
                tensor=video[None],
                save_file=save_file,
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1)
            )
            
            logging.info(f"✓ 成功生成视频: {save_file}")
            
            # 清理显存
            del video
            torch.cuda.synchronize()
            
        except Exception as e:
            logging.error(f"✗ 生成失败 (Index {prompt_data['index']}): {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    logging.info(f"{'='*60}")
    logging.info("批量生成完成！")
    logging.info(f"输出目录: {output_dir}")
    logging.info(f"{'='*60}")
    
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="优化版批量生成视频")
    parser.add_argument("--json_file", type=str, required=True, help="JSON文件路径")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="模型checkpoint目录")
    parser.add_argument("--task", type=str, default='t2v-A14B', help="任务类型")
    parser.add_argument("--size", type=str, default='1280*720', help="视频尺寸")
    parser.add_argument("--output_dir", type=str, default='./output_videos', help="输出目录")
    parser.add_argument("--start_index", type=int, default=None, help="起始index")
    parser.add_argument("--end_index", type=int, default=None, help="结束index（不包含）")
    parser.add_argument("--categories", type=str, nargs='+', default=None, help="类别列表")
    parser.add_argument("--offload_model", action="store_true", default=True, help="是否offload模型")
    parser.add_argument("--convert_model_dtype", action="store_true", default=True, help="是否转换模型数据类型")
    parser.add_argument("--frame_num", type=int, default=None, help="帧数")
    parser.add_argument("--sample_steps", type=int, default=20, help="采样步数")
    parser.add_argument("--sample_shift", type=float, default=None, help="采样shift")
    parser.add_argument("--sample_guide_scale", type=float, default=None, help="引导scale")
    parser.add_argument("--base_seed", type=int, default=-1, help="基础随机种子")
    
    args = parser.parse_args()
    
    batch_generate_optimized(
        json_file=args.json_file,
        ckpt_dir=args.ckpt_dir,
        task=args.task,
        size=args.size,
        output_dir=args.output_dir,
        start_index=args.start_index,
        end_index=args.end_index,
        categories=args.categories,
        offload_model=args.offload_model,
        convert_model_dtype=args.convert_model_dtype,
        frame_num=args.frame_num,
        sample_steps=args.sample_steps,
        sample_shift=args.sample_shift,
        sample_guide_scale=args.sample_guide_scale,
        base_seed=args.base_seed
    )
