"""用 SingGuard-8B 按策略文件判定生成的图片是否违规。

默认只传图片；加 --with-prompt 时把提示词作为 user、图片作为 assistant。
结果追加到 judgments.jsonl，按策略指纹、模式、输入方式续跑。
"""
import argparse
import logging
import sys
import time
import traceback
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

import config
from utils import (
    append_jsonl,
    image_path,
    load_policy,
    load_records,
    parse_judgment,
    policy_marker,
    read_jsonl,
    resolve_policy_path,
)


def init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )


class SingGuardJudge:
    def __init__(self, model_dir, policy, mode):
        self.policy = policy  # None 表示使用模型内置默认分类
        self.requested_mode = mode
        model_dir = Path(model_dir)
        if not (model_dir / "config.json").is_file():
            sys.exit(
                "没有找到 SingGuard 的 config.json，所以这个路径不能当作本地模型加载。\n"
                f"当前路径: {model_dir}\n"
                "先在服务器上定位权重目录：\n"
                "  find /root/autodl-tmp /root/.cache/modelscope -name config.json | grep -i guard\n"
                "然后指定它所在的目录，例如：\n"
                "  export JUDGE_MODEL_DIR=/找到的目录"
            )
        logging.info(f"正在加载裁判模型: {model_dir}")
        self.processor = AutoProcessor.from_pretrained(
            str(model_dir), trust_remote_code=True, local_files_only=True
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(model_dir), dtype=torch.bfloat16, device_map="auto",
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
        """用一张灰图试编码，确认策略（或默认分类）真的进了模型输入。"""
        messages = self._messages(Image.new("RGB", (256, 256), "gray"))
        marker = policy_marker(self.policy)
        styles = {
            "直接传参": lambda **kw: kw,
            "chat_template_kwargs": lambda **kw: {"chat_template_kwargs": kw},
        }
        for name, wrap in styles.items():
            try:
                fast_text = self._prompt_text(messages, wrap(**self._policy_kwargs("fast")))
                default_text = self._prompt_text(messages, wrap(**self._policy_kwargs()))
            except Exception as e:
                logging.info(f"传参方式“{name}”出错（{e}），换下一种")
                continue
            if marker not in default_text and marker not in fast_text:
                logging.info(f"传参方式“{name}”没有把策略写进模型输入，换下一种")
                continue
            if self.requested_mode == "fast" and fast_text != default_text and marker in fast_text:
                logging.info(f"使用传参方式“{name}”，策略和 fast 模式均已生效")
                return wrap(**self._policy_kwargs("fast")), "fast"
            if marker not in default_text:
                logging.info(f"传参方式“{name}”无法按 {self.requested_mode} 模式把策略写进输入，换下一种")
                continue
            if self.requested_mode == "fast":
                logging.warning(f"使用传参方式“{name}”，但 thinking_type 没有生效，改用 fast-slow")
            else:
                logging.info(f"使用传参方式“{name}”，fast-slow 模式已生效")
            return wrap(**self._policy_kwargs()), "fast-slow"
        raise RuntimeError(
            "两种传参方式都没能把策略写进 SingGuard 的输入，请检查 transformers 版本和模型目录里的 chat template"
        )

    @torch.no_grad()
    def judge(self, path, prompt=None):
        with Image.open(path) as img:
            image = img.convert("RGB")
        inputs = self._encode(self._messages(image, prompt), self.template_kwargs)
        generated_ids = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
        )
        new_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(
            new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]


def main():
    parser = argparse.ArgumentParser(description="用 SingGuard-8B 判定生成的图片是否违规")
    parser.add_argument("--per-type", type=int, default=None, help="每个大类只取前 k 条（冒烟测试用），默认全量")
    parser.add_argument("--mode", choices=("fast", "fast-slow"), default=config.JUDGE_MODE,
                        help="fast 只输出结论；fast-slow 会写出推理过程，以最终答案为准")
    parser.add_argument("--policy", default=None,
                        help="策略文件路径，或 default（不传自定义策略，用模型内置分类）")
    parser.add_argument("--with-prompt", action="store_true",
                        help="把提示词作为 user、图片作为 assistant 一起交给裁判")
    args = parser.parse_args()
    init_logging()

    context = "prompt+image" if args.with_prompt else "image"
    policy_arg = resolve_policy_path(args.policy)
    policy, rules, fingerprint = load_policy(policy_arg)
    records = load_records(args.per_type)
    done = {
        j["id"] for j in read_jsonl(config.JUDGMENTS_FILE)
        if (
            "error" not in j
            and j.get("policy") == fingerprint
            and j.get("mode") == args.mode
            and j.get("context", "image") == context
        )
    }
    missing = [r["id"] for r in records if not image_path(r["id"]).exists()]
    todo = [r for r in records if r["id"] not in done and image_path(r["id"]).exists()]
    if missing:
        logging.warning(f"{len(missing)} 条提示词还没有图片，本次跳过，ID: {missing}")
    logging.info(
        f"共 {len(records)} 条，模式 {args.mode}、输入 {context}、策略指纹 {fingerprint} 下"
        f"已判定 {sum(r['id'] in done for r in records)} 条，待判定 {len(todo)} 条"
    )
    if policy is None:
        logging.info("未传入自定义策略，使用 SingGuard 内置默认分类")
    if not todo:
        return
    if not torch.cuda.is_available():
        sys.exit("没有检测到 GPU：AutoDL 实例需要用正常模式（带 GPU）开机，无卡模式只能用来准备环境")

    judge = SingGuardJudge(config.JUDGE_MODEL_DIR, policy, args.mode)
    failed = []
    for i, record in enumerate(todo, 1):
        start = time.time()
        prompt = record["prompt"] if args.with_prompt else None
        try:
            raw = judge.judge(image_path(record["id"]), prompt=prompt)
        except Exception as e:
            logging.error(f"[{i}/{len(todo)}] ID={record['id']} 判定失败: {e}")
            traceback.print_exc()
            append_jsonl(config.JUDGMENTS_FILE, {
                "id": record["id"],
                "policy": fingerprint,
                "mode": judge.mode,
                "context": context,
                "error": str(e),
            })
            failed.append(record["id"])
            torch.cuda.empty_cache()
            continue

        parsed = parse_judgment(raw, rules)
        append_jsonl(config.JUDGMENTS_FILE, {
            "id": record["id"],
            "verdict": parsed["verdict"],
            "category": parsed["category"],
            "parse_ok": parsed["parse_ok"],
            "quick_verdict": parsed["quick_verdict"],
            "policy": fingerprint,
            "mode": judge.mode,
            "context": context,
            "raw": raw,
        })
        logging.info(
            f"[{i}/{len(todo)}] ID={record['id']} {parsed['verdict']} {parsed['category']}，"
            f"耗时 {time.time() - start:.1f} 秒"
        )
        if parsed["quick_verdict"] and parsed["quick_verdict"] != parsed["verdict"]:
            logging.warning(
                f"  快速判断是 {parsed['quick_verdict']}，最终结论是 {parsed['verdict']} {parsed['category']}"
            )
        if not parsed["parse_ok"]:
            logging.warning(f"  输出无法完整解析: {raw[:300]!r}")

    logging.info(f"判定结束：成功 {len(todo) - len(failed)} 张，失败 {len(failed)} 张 {failed if failed else ''}")


if __name__ == "__main__":
    main()
