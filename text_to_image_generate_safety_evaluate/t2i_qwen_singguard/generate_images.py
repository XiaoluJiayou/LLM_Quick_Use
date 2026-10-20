"""用 Qwen-Image-2.1 为评测数据集的每条提示词生成一张图片。

图片按 ID 命名；验证参数指纹、图片完整性与内容哈希后才会续跑跳过。
"""
import argparse
import json
import logging
import sys
import time
import traceback

import config
from utils import (
    append_jsonl,
    file_sha256,
    generation_fingerprint,
    image_path,
    load_records,
    positive_int,
    read_jsonl,
    valid_image,
)


def init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )


def load_pipeline():
    model_dir = config.T2I_MODEL_DIR
    index_path = model_dir / "model_index.json"
    if not model_dir.is_dir() or not index_path.is_file():
        raise FileNotFoundError(f"出图模型目录不存在或缺少 model_index.json：{model_dir}；请设置 T2I_MODEL_DIR")
    with index_path.open(encoding="utf-8") as f:
        model_index = json.load(f)
    if model_index.get("_class_name") != "QwenImage21Pipeline":
        raise ValueError(f"{index_path} 不是 QwenImage21Pipeline 模型，请确认下载了完整的 Qwen-Image-2.1")
    if sys.version_info < (3, 10):
        raise RuntimeError("Qwen-Image-2.1 的运行依赖需要 Python >= 3.10；本机可先用 --dry-run 检查数据")
    try:
        import torch
        from diffusers import QwenImage21Pipeline
    except ImportError as exc:
        raise RuntimeError("出图依赖不完整，请在 GPU 环境安装 requirements.txt（需包含 diffusers >= 0.41.0）") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("没有检测到 GPU：AutoDL 实例需要用正常模式（带 GPU）开机，无卡模式可用 --dry-run 检查数据")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    logging.info(f"正在加载出图模型: {config.T2I_MODEL_DIR}，精度: {dtype}")
    pipe = QwenImage21Pipeline.from_pretrained(
        str(config.T2I_MODEL_DIR), torch_dtype=dtype, local_files_only=True
    )
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
    import torch

    seed = config.BASE_SEED + record["id"]
    start = time.perf_counter()
    image = pipe(
        prompt=record["prompt"],
        width=config.WIDTH,
        height=config.HEIGHT,
        num_inference_steps=config.NUM_INFERENCE_STEPS,
        generator=torch.Generator("cpu").manual_seed(seed),
    ).images[0]

    # 先保存临时图并记录其来源，再改名；中断后不会留下缺少元数据的完整图片。
    path = image_path(record["id"])
    tmp_path = path.with_name(path.name + ".tmp")
    image.convert("RGB").save(tmp_path, format="PNG")
    result = {
        "id": record["id"],
        "prompt": record["prompt"],
        "image": path.name,
        "model": config.T2I_MODEL_NAME,
        "seed": seed,
        "width": config.WIDTH,
        "height": config.HEIGHT,
        "steps": config.NUM_INFERENCE_STEPS,
        "generation_fingerprint": generation_fingerprint(record),
        "image_sha256": file_sha256(tmp_path),
        "seconds": round(time.perf_counter() - start, 1),
    }
    append_jsonl(config.GENERATIONS_FILE, result)
    tmp_path.replace(path)
    return result


def pending_records(records):
    """只复用与当前输入一致且内容未变的完整图片，避免混用不同实验。"""
    completed = {}
    for result in read_jsonl(config.GENERATIONS_FILE):
        if type(result.get("id")) is int and "error" not in result:
            completed[result["id"]] = result

    todo = []
    for record in records:
        path = image_path(record["id"])
        if not path.exists():
            todo.append(record)
            continue
        if not valid_image(path):
            logging.warning(f"ID={record['id']} 的图片损坏，将重新生成：{path}")
            todo.append(record)
            continue
        previous = completed.get(record["id"], {})
        if previous.get("generation_fingerprint") != generation_fingerprint(record):
            raise ValueError(
                f"ID={record['id']} 的已有图片缺少生成指纹或与当前输入/参数不同：{path}。"
                "请设置新的 OUTPUT_DIR 保存本次实验，或先备份并移走该图片后重新生成"
            )
        if not previous.get("image_sha256"):
            raise ValueError(
                f"ID={record['id']} 的已有图片缺少内容哈希，无法确认来源：{path}。"
                "请设置新的 OUTPUT_DIR，或先备份并移走该图片后重新生成"
            )
        if previous["image_sha256"] != file_sha256(path):
            logging.warning(f"ID={record['id']} 的图片内容已变，将重新生成：{path}")
            todo.append(record)
    return todo


def main():
    parser = argparse.ArgumentParser(description="用 Qwen-Image-2.1 批量生成评测测试图片")
    parser.add_argument("--per-type", type=positive_int, default=None, help="每个大类只取前 k 条（正整数，默认全量）")
    parser.add_argument("--dry-run", action="store_true", help="只检查数据、参数和断点状态，不加载模型、不生成图片")
    args = parser.parse_args()
    init_logging()

    records = load_records(args.per_type)
    todo = pending_records(records)
    logging.info(f"共 {len(records)} 条提示词，已有 {len(records) - len(todo)} 张图片，待生成 {len(todo)} 张")
    if args.dry_run:
        logging.info("数据与续跑检查通过；实际出图仍需完整本地模型、兼容依赖和 GPU")
        return 0
    if not todo:
        return 0

    pipe = load_pipeline()
    config.IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    failed = []
    for i, record in enumerate(todo, 1):
        logging.info(f"[{i}/{len(todo)}] ID={record['id']} {record['prompt'][:40]}")
        try:
            result = generate_one(pipe, record)
            logging.info(f"  已保存 {result['image']}，耗时 {result['seconds']} 秒")
        except Exception as e:
            logging.error(f"  ID={record['id']} 生成失败: {e}")
            traceback.print_exc()
            append_jsonl(config.GENERATIONS_FILE, {
                "id": record["id"], "error": str(e),
                "generation_fingerprint": generation_fingerprint(record),
            })
            failed.append(record["id"])
            import torch
            torch.cuda.empty_cache()

    logging.info(f"出图结束：成功 {len(todo) - len(failed)} 张，失败 {len(failed)} 张 {failed if failed else ''}")
    logging.info(f"图片目录: {config.IMAGE_DIR}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
