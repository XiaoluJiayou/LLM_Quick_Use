import os
import argparse
import datetime
import json
import random
import sys
import logging
import PIL.Image
import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, UMT5EncoderModel
from torchvision.io import write_video
from longcat_video.pipeline_longcat_video import LongCatVideoPipeline
from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
from longcat_video.context_parallel import context_parallel_util
from longcat_video.context_parallel.context_parallel_util import init_context_parallel


def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


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
            handlers=[logging.StreamHandler(stream=sys.stdout)]
        )
    else:
        logging.basicConfig(level=logging.ERROR)


def generate(args):
    # 分布式环境初始化
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    local_rank = rank % num_gpus
    torch.cuda.set_device(local_rank)

    init_logging(rank)

    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600 * 24))
    global_rank = dist.get_rank()
    num_processes = dist.get_world_size()

    # 在加载模型前初始化 context parallel
    init_context_parallel(context_parallel_size=args.context_parallel_size, global_rank=global_rank,
                          world_size=num_processes)
    cp_size = context_parallel_util.get_cp_size()
    cp_split_hw = context_parallel_util.get_optimal_split(cp_size)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载并筛选 prompts
    prompts = load_prompts(args.json_file)
    logging.info(f"总共加载了 {len(prompts)} 个prompts")

    if args.categories:
        prompts = [p for p in prompts if p.get('category') in args.categories]
        logging.info(f"筛选出 {len(prompts)} 个指定类别的prompts")

    if args.start_index is not None or args.end_index is not None:
        filtered_prompts = []
        for p in prompts:
            idx = p['index']
            if args.start_index is not None and idx < args.start_index:
                continue
            if args.end_index is not None and idx >= args.end_index:
                continue
            filtered_prompts.append(p)
        prompts = filtered_prompts
        logging.info(f"筛选出 {len(prompts)} 个指定index范围的prompts")

    # ========================================
    # 关键步骤：只加载一次模型！
    # ========================================
    logging.info("=" * 60)
    logging.info("正在加载模型（只加载一次）...")
    logging.info("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir, subfolder="tokenizer", torch_dtype=torch.bfloat16)
    text_encoder = UMT5EncoderModel.from_pretrained(args.checkpoint_dir, subfolder="text_encoder",
                                                    torch_dtype=torch.bfloat16)
    vae = AutoencoderKLWan.from_pretrained(args.checkpoint_dir, subfolder="vae", torch_dtype=torch.bfloat16)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.checkpoint_dir, subfolder="scheduler",
                                                                torch_dtype=torch.bfloat16)
    dit = LongCatVideoTransformer3DModel.from_pretrained(args.checkpoint_dir, subfolder="dit", cp_split_hw=cp_split_hw,
                                                         torch_dtype=torch.bfloat16)

    if args.enable_compile:
        dit = torch.compile(dit)

    pipe = LongCatVideoPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=scheduler,
        dit=dit,
    )
    pipe.to(local_rank)
    logging.info("✓ 模型加载完成！")
    logging.info("=" * 60)

    # 处理随机种子同步：确保多卡情况下同一个 prompt 的随机种子一致
    base_seed = args.base_seed
    if base_seed < 0:
        if global_rank == 0:
            base_seed = random.randint(0, sys.maxsize)
        base_seed_list = [base_seed]
        dist.broadcast_object_list(base_seed_list, src=0)
        base_seed = base_seed_list[0]

    # ========================================
    # 批量生成视频
    # ========================================
    for i, prompt_data in enumerate(prompts):
        logging.info(f"{'=' * 60}")
        logging.info(f"正在生成第 {i + 1}/{len(prompts)} 个视频")
        logging.info(f"Index: {prompt_data['index']}")
        logging.info(f"Category: {prompt_data.get('category', 'N/A')}")
        logging.info(f"Prompt: {prompt_data['prompt'][:80]}...")
        logging.info(f"{'=' * 60}")

        try:
            # 依据 index 生成确定的随机种子，保证多卡生成结果一致
            seed = base_seed + prompt_data['index']
            generator = torch.Generator(device=local_rank)
            generator.manual_seed(seed)

            # 生成视频 (去除了 distill 和 refine 流程，专注于基础 t2v)
            output = pipe.generate_t2v(
                prompt=prompt_data['prompt'],
                negative_prompt=args.negative_prompt,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            )[0]

            # 保存视频 (只在主节点保存，避免多卡重复写入)
            if global_rank == 0:
                output_tensor = torch.from_numpy(np.array(output))
                output_tensor = (output_tensor * 255).clamp(0, 255).to(torch.uint8)

                save_file = os.path.join(args.output_dir, f"{prompt_data['index']}.mp4")
                logging.info(f"保存视频到: {save_file}")
                write_video(
                    save_file,
                    output_tensor,
                    fps=args.fps,
                    video_codec="libx264",
                    options={"crf": f"{args.crf}"}
                )
                logging.info(f"✓ 成功生成视频: {save_file}")

            # 清理当前循环的显存
            del output
            torch_gc()

        except Exception as e:
            logging.error(f"✗ 生成失败 (Index {prompt_data['index']}): {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    logging.info(f"{'=' * 60}")
    logging.info("批量生成完成！")
    logging.info(f"输出目录: {args.output_dir}")
    logging.info(f"{'=' * 60}")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _parse_args():
    parser = argparse.ArgumentParser(description="批量生成视频脚本")

    # 数据与输出相关参数
    parser.add_argument("--json_file", type=str, required=True,
                        help="包含prompts的JSON文件路径 (如 T2VSafetyBench_sampled_3class.json)")
    parser.add_argument("--output_dir", type=str, default='./output_videos', help="视频输出目录")
    parser.add_argument("--start_index", type=int, default=None, help="起始index")
    parser.add_argument("--end_index", type=int, default=None, help="结束index（不包含）")
    parser.add_argument("--categories", type=str, nargs='+', default=None,
                        help="要生成的类别列表，如 Pornography Violence")

    # 模型相关参数
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="模型checkpoint目录")
    parser.add_argument("--context_parallel_size", type=int, default=1, help="上下文并行大小")
    parser.add_argument("--enable_compile", action="store_true", help="是否使用torch.compile加速")

    # 生成相关参数
    parser.add_argument("--height", type=int, default=480, help="视频高度")
    parser.add_argument("--width", type=int, default=832, help="视频宽度")
    parser.add_argument("--num_frames", type=int, default=93, help="生成视频帧数")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="推理步数")
    parser.add_argument("--guidance_scale", type=float, default=4.0, help="引导系数")
    parser.add_argument("--fps", type=int, default=15, help="输出视频帧率")
    parser.add_argument("--crf", type=int, default=18, help="视频编码质量 (越小质量越高)")
    parser.add_argument("--base_seed", type=int, default=-1, help="基础随机种子，-1表示完全随机")

    parser.add_argument("--negative_prompt", type=str,
                        default="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
                        help="负面提示词")

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
