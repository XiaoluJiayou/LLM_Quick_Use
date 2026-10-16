"""用 Qwen-Image-2.1 为 JADE-T2I 的每条提示词生成一张图片。

图片按 ID 命名，已存在的会跳过，所以中断后直接重新运行即可接着生成。
"""
import argparse
import logging
import sys
import time
import traceback

import torch
from diffusers import QwenImage21Pipeline

import config
from utils import append_jsonl, image_path, load_records


def init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )


def load_pipeline():
    if not torch.cuda.is_available():
        sys.exit("没有检测到 GPU：AutoDL 实例需要用正常模式（带 GPU）开机，无卡模式只能用来准备环境")

    logging.info(f"正在加载出图模型: {config.T2I_MODEL_DIR}")
    pipe = QwenImage21Pipeline.from_pretrained(str(config.T2I_MODEL_DIR), torch_dtype=torch.bfloat16)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    if total_gb < config.OFFLOAD_BELOW_GB:
        logging.info(f"显存 {total_gb:.0f}GB，启用 CPU 卸载")
        pipe.enable_model_cpu_offload()
    else:
        logging.info(f"显存 {total_gb:.0f}GB，整个模型放到 GPU 上")
        pipe.to("cuda")
    # 每张图结束时已有一行带耗时的日志，关掉逐步进度条，免得刷满日志文件
    pipe.set_progress_bar_config(disable=True)
    logging.info(f"模型加载完成，精度: {pipe.dtype}")
    return pipe


def generate_one(pipe, record):
    seed = config.BASE_SEED + record["id"]
    start = time.time()
    image = pipe(
        prompt=record["prompt"],
        width=config.WIDTH,
        height=config.HEIGHT,
        num_inference_steps=config.NUM_INFERENCE_STEPS,
        generator=torch.Generator("cpu").manual_seed(seed),
    ).images[0]

    # 先写临时文件再改名，避免中断时留下半张图被当成“已生成”
    path = image_path(record["id"])
    tmp_path = path.with_name(path.name + ".tmp")
    image.convert("RGB").save(tmp_path, format="PNG")
    tmp_path.replace(path)

    return {
        "id": record["id"],
        "prompt": record["prompt"],
        "image": path.name,
        "model": config.T2I_MODEL_NAME,
        "seed": seed,
        "width": config.WIDTH,
        "height": config.HEIGHT,
        "steps": config.NUM_INFERENCE_STEPS,
        "seconds": round(time.time() - start, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="用 Qwen-Image-2.1 批量生成 JADE-T2I 测试图片")
    parser.add_argument("--per-type", type=int, default=None, help="每个大类只取前 k 条（冒烟测试用），默认全量")
    args = parser.parse_args()
    init_logging()

    records = load_records(args.per_type)
    config.IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    todo = [r for r in records if not image_path(r["id"]).exists()]
    logging.info(f"共 {len(records)} 条提示词，已有 {len(records) - len(todo)} 张图片，待生成 {len(todo)} 张")
    if not todo:
        return

    pipe = load_pipeline()
    failed = []
    for i, record in enumerate(todo, 1):
        logging.info(f"[{i}/{len(todo)}] ID={record['id']} {record['prompt'][:40]}")
        try:
            result = generate_one(pipe, record)
            append_jsonl(config.GENERATIONS_FILE, result)
            logging.info(f"  已保存 {result['image']}，耗时 {result['seconds']} 秒")
        except Exception as e:
            logging.error(f"  ID={record['id']} 生成失败: {e}")
            traceback.print_exc()
            append_jsonl(config.GENERATIONS_FILE, {"id": record["id"], "error": str(e)})
            failed.append(record["id"])
            torch.cuda.empty_cache()

    logging.info(f"出图结束：成功 {len(todo) - len(failed)} 张，失败 {len(failed)} 张 {failed if failed else ''}")
    logging.info(f"图片目录: {config.IMAGE_DIR}")


if __name__ == "__main__":
    main()
