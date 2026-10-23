"""用裁判模型按策略文件判定生成的图片是否违规。

默认只传图片；加 --with-prompt 时把提示词作为 user、图片作为 assistant。
结果追加到 judgments.jsonl，按策略指纹、模式、输入方式续跑。
"""
import argparse
import logging
import re
import sys
import time
import traceback
from pathlib import Path

import config
from utils import (
    append_jsonl,
    file_sha256,
    generation_matches,
    image_path,
    latest_matching_judgments,
    load_generation_metadata,
    load_policy,
    load_records,
    parse_judgment,
    policy_marker,
    positive_int,
    record_fingerprint,
    resolve_policy_path,
    valid_image,
)


def init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )


class ImageJudge:
    def __init__(self, model_dir, policy, mode):
        import torch
        from PIL import Image
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.image_class = Image
        self.policy = policy  # None 表示使用模型内置默认分类
        self.requested_mode = mode
        model_dir = Path(model_dir)
        if not (model_dir / "config.json").is_file():
            raise FileNotFoundError(
                "没有找到裁判模型的 config.json，所以这个路径不能当作本地模型加载。\n"
                f"当前路径: {model_dir}\n"
                "先在服务器上定位权重目录：\n"
                "  find /root/autodl-tmp /root/.cache/modelscope -name config.json | grep -i judge\n"
                "然后指定它所在的目录，例如：\n"
                "  export JUDGE_MODEL_DIR=/找到的目录"
            )
        logging.info(f"正在加载裁判模型: {model_dir}")
        self.processor = AutoProcessor.from_pretrained(
            str(model_dir), trust_remote_code=True, local_files_only=True
        )
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(model_dir), dtype=dtype, device_map="auto",
            trust_remote_code=True, local_files_only=True,
        ).eval()
        self.template_kwargs, self.mode = self._pick_template_kwargs()
        self.max_new_tokens = (
            config.JUDGE_MAX_NEW_TOKENS if self.mode == "fast" else config.JUDGE_SLOW_MAX_NEW_TOKENS
        )
        logging.info(f"裁判模型加载完成，模式: {self.mode}，最多生成 {self.max_new_tokens} 个 token")

    @staticmethod
    def _messages(image, prompt=None):
        if prompt:
            return [
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
                {"role": "assistant", "content": [{"type": "image", "image": image}]},
            ]
        return [{"role": "user", "content": [{"type": "image", "image": image}]}]

    def _policy_kwargs(self, thinking_type=None):
        kwargs = {}
        if self.policy is not None:
            kwargs["policy"] = self.policy
        if thinking_type is not None:
            kwargs["thinking_type"] = thinking_type
        return kwargs

    def _encode(self, messages, template_kwargs):
        return self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            **template_kwargs,
        ).to(self.model.device)

    def _prompt_text(self, messages, template_kwargs):
        input_ids = self._encode(messages, template_kwargs)["input_ids"][0]
        return self.processor.tokenizer.decode(
            input_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

    def _pick_template_kwargs(self):
        """验证实际编码中的完整策略及请求模式，不静默切换推理模式。"""
        messages = self._messages(self.image_class.new("RGB", (256, 256), "gray"))
        expected_policy = self.policy or policy_marker(None)
        mode_pattern = re.compile(
            rf"<thinking_type>\s*{re.escape(self.requested_mode)}\s*</thinking_type>"
        )
        styles = {
            "直接传参": lambda **kw: kw,
            "chat_template_kwargs": lambda **kw: {"chat_template_kwargs": kw},
        }
        for name, wrap in styles.items():
            try:
                kwargs = wrap(**self._policy_kwargs(self.requested_mode))
                prompt_text = self._prompt_text(messages, kwargs)
            except Exception as e:
                logging.info(f"传参方式“{name}”出错（{e}），换下一种")
                continue
            if expected_policy not in prompt_text:
                logging.info(f"传参方式“{name}”没有把策略写进模型输入，换下一种")
                continue
            if not mode_pattern.search(prompt_text):
                logging.info(f"传参方式“{name}”没有启用 {self.requested_mode} 模式，换下一种")
                continue
            logging.info(f"使用传参方式“{name}”，策略和 {self.requested_mode} 模式均已生效")
            return kwargs, self.requested_mode
        raise RuntimeError(
            f"两种传参方式都没能同时启用完整策略和 {self.requested_mode} 模式；"
            "请检查 transformers 版本和模型目录里的 chat_template.jinja，不会自动切换模式"
        )

    def judge(self, path, prompt=None, max_new_tokens=None):
        with self.image_class.open(path) as img:
            image = img.convert("RGB")
        inputs = self._encode(self._messages(image, prompt), self.template_kwargs)
        limit = self.max_new_tokens if max_new_tokens is None else max_new_tokens
        with self.torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=limit, do_sample=False
            )
        new_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(
            new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]


def main():
    parser = argparse.ArgumentParser(description="用裁判模型判定生成的图片是否违规")
    parser.add_argument("--per-type", type=positive_int, default=None, help="每个大类只取前 k 条（小样本试跑），默认全量")
    parser.add_argument("--mode", choices=("fast", "fast-slow"), default=config.JUDGE_MODE,
                        help="默认 fast-slow（写出推理，以最终答案为准）；fast 只输出结论")
    parser.add_argument("--policy", default=None,
                        help="策略文件路径，或 default（不传自定义策略，用模型内置分类）")
    parser.add_argument("--with-prompt", action="store_true",
                        help="把提示词作为 user、图片作为 assistant 一起交给裁判")
    parser.add_argument("--dry-run", action="store_true",
                        help="检查数据、策略和现有图片，预览待办，不要求模型或 GPU")
    args = parser.parse_args()
    init_logging()

    context = "prompt+image" if args.with_prompt else "image"
    policy_arg = resolve_policy_path(args.policy)
    policy, rules, fingerprint = load_policy(policy_arg)
    records = load_records(args.per_type)
    generation_metadata = load_generation_metadata()
    latest, stale_count = latest_matching_judgments(records, fingerprint, args.mode, context)
    done = {record_id for record_id, item in latest.items() if item.get("parse_ok") is True}
    available = {
        r["id"] for r in records
        if valid_image(image_path(r["id"])) and generation_matches(r, generation_metadata.get(r["id"]))
    }
    missing = [r["id"] for r in records if r["id"] not in available]
    todo = [r for r in records if r["id"] not in done and r["id"] in available]
    if missing:
        message = (
            f"{len(missing)} 条提示词的图片缺失、损坏或生成来源/参数不匹配，"
            f"需先运行出图阶段，ID: {missing}"
        )
        if args.dry_run:
            logging.info(message)
        else:
            logging.error(message)
    if stale_count:
        logging.info(f"已忽略 {stale_count} 条旧输入、旧图片、旧模型或解析失败的判定")
    logging.info(
        f"共 {len(records)} 条，模式 {args.mode}、输入 {context}、策略指纹 {fingerprint} 下"
        f"已判定 {sum(r['id'] in done for r in records)} 条，待判定 {len(todo)} 条"
    )
    if policy is None:
        logging.info("未传入自定义策略，使用裁判模型内置默认分类")
    if args.dry_run:
        logging.info(f"预览完成：已有图片待判定 {len(todo)} 张，补全图片后待判定 {len(missing)} 张")
        logging.info("未验证 GPU、模型权重或实际推理；实际运行前需先补全图片")
        return 0
    if not todo:
        return 1 if missing else 0
    import torch

    if not torch.cuda.is_available():
        sys.exit("没有检测到 GPU：AutoDL 实例需要用正常模式（带 GPU）开机，无卡模式只能用来准备环境")

    judge = ImageJudge(config.JUDGE_MODEL_DIR, policy, args.mode)
    hard_failed = []
    unparsed = []
    revised_unsafe_to_safe = 0
    revised_other = 0
    for i, record in enumerate(todo, 1):
        start = time.time()
        prompt = record["prompt"] if args.with_prompt else None
        metadata = {
            "id": record["id"],
            "record_fingerprint": record_fingerprint(record),
            "judge_model": str(config.JUDGE_MODEL_DIR.resolve()),
            "policy": fingerprint,
            "mode": judge.mode,
            "context": context,
        }
        try:
            path = image_path(record["id"])
            metadata["image_sha256"] = file_sha256(path)
            raw = judge.judge(path, prompt=prompt)
        except Exception as e:
            logging.error(f"[{i}/{len(todo)}] ID={record['id']} 判定失败: {e}")
            traceback.print_exc()
            append_jsonl(config.JUDGMENTS_FILE, {
                **metadata,
                "error": str(e),
            })
            hard_failed.append(record["id"])
            torch.cuda.empty_cache()
            continue

        parsed = parse_judgment(raw, rules)
        retry_limit = config.JUDGE_SLOW_RETRY_MAX_NEW_TOKENS
        if judge.mode == "fast-slow" and not parsed["parse_ok"] and retry_limit > judge.max_new_tokens:
            logging.warning(
                f"  ID={record['id']} 未写出完整答案，用 {retry_limit} 个 token 再判一次"
            )
            try:
                raw = judge.judge(path, prompt=prompt, max_new_tokens=retry_limit)
            except Exception as e:
                logging.error(f"[{i}/{len(todo)}] ID={record['id']} 再判失败: {e}")
                traceback.print_exc()
                append_jsonl(config.JUDGMENTS_FILE, {**metadata, "error": str(e)})
                hard_failed.append(record["id"])
                torch.cuda.empty_cache()
                continue
            parsed = parse_judgment(raw, rules)

        append_jsonl(config.JUDGMENTS_FILE, {
            **metadata,
            "verdict": parsed["verdict"],
            "category": parsed["category"],
            "parse_ok": parsed["parse_ok"],
            "quick_verdict": parsed["quick_verdict"],
            "raw": raw,
        })
        logging.info(
            f"[{i}/{len(todo)}] ID={record['id']} {parsed['verdict']} {parsed['category']}，"
            f"耗时 {time.time() - start:.1f} 秒"
        )
        if parsed["parse_ok"] and parsed["quick_verdict"] and parsed["quick_verdict"] != parsed["verdict"]:
            if parsed["quick_verdict"] == "unsafe" and parsed["verdict"] == "safe":
                revised_unsafe_to_safe += 1
            else:
                revised_other += 1
        if not parsed["parse_ok"]:
            unparsed.append(record["id"])
            logging.warning(f"  输出无法完整解析: {raw[:300]!r}")

    if revised_unsafe_to_safe:
        logging.info(f"{revised_unsafe_to_safe} 张由快速违规改判为安全")
    if revised_other:
        logging.info(f"{revised_other} 张快速结论与最终结论不一致")
    logging.info(
        f"判定结束：成功 {len(todo) - len(hard_failed) - len(unparsed)} 张，"
        f"解析失败 {len(unparsed)} 张 {unparsed if unparsed else ''}，"
        f"判定异常 {len(hard_failed)} 张 {hard_failed if hard_failed else ''}"
    )
    # 解析失败仍留给报告统计；缺图或推理异常才中断后续汇总
    return 1 if hard_failed or missing else 0


if __name__ == "__main__":
    sys.exit(main())
