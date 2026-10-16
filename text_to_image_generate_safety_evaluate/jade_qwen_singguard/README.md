# JADE + Qwen-Image-2.1 + SingGuard 文生图安全评测

用 [JADE-T2I](https://github.com/whitzard-ai/jade-db/tree/main/jade-t2i-v1.0) 的中文提示词（227 条），在本地开源模型上完成三步：

1. **Qwen-Image-2.1** 按提示词出图
2. **SingGuard-8B** 按 `policy_jade_t2i.md`（GB/T 45654-2025 附录 A 五大类）判定图片是否违规
3. 汇总违规率、命中率、解析失败率，写出 `report.csv`

代码在本机编写，出图和判定在 AutoDL（有 GPU）上运行。数据集和生成图片含违规内容，`data/`、`outputs/` 已在仓库根目录 `.gitignore` 中忽略，不要提交。

## 目录

```text
jade_qwen_singguard/
├── config.py              # 路径和出图参数，可用同名环境变量覆盖
├── utils.py               # 读 CSV、读写 jsonl、策略解析、JADE→国标对照
├── generate_images.py     # 出图
├── judge_singguard.py     # 判定
├── report.py              # 汇总报告
├── policy_jade_t2i.md     # GB/T 45654-2025 附录 A 五大类（31 小类作说明）
├── requirements.txt
├── data/jade_benchmark_zh.csv
└── scripts/run_demo.sh    # 依次跑出图、判定、报告
```

默认模型目录（可在 `config.py` 里改）。ModelScope 缓存目录里的 `snapshots/master` 会自动识别：

```text
/root/autodl-tmp/models/Qwen-Image-2.1
/root/autodl-tmp/models/Sing-Guard-8b
```

结果写到 `outputs/Qwen-Image-2.1/`：

| 文件 | 内容 |
|---|---|
| `images/{id}.png` | 生成图片 |
| `generations.jsonl` | 每张图的种子、尺寸、步数、耗时 |
| `judgments.jsonl` | 判定原文和解析结果 |
| `report.csv` | 汇总表，可用 Excel 打开 |

## 环境

torch、CUDA 用 AutoDL 镜像自带的（torch ≥ 2.4）。其余：

```bash
pip install "transformers>=5.17" accelerate modelscope pillow
source /etc/network_turbo
pip install "git+https://github.com/huggingface/diffusers.git"
unset http_proxy https_proxy
```

`diffusers` 需要带 `QwenImage21Pipeline` 的版本（GitHub 最新版，或 ≥ 0.41.0）。装完检查：

```bash
python -c "from diffusers import QwenImage21Pipeline; from transformers import AutoModelForImageTextToText; print('ok')"
```

模型用 ModelScope 下载到上面的默认目录即可，仓库 ID：

- `Qwen/Qwen-Image-2.1`
- `inclusionAI/Sing-Guard-8b`（若没有，试 `inclusionAI/SingGuard-8b`）

两个模型大约 50GB，数据盘建议留到 80GB 左右。

## 运行

在项目目录下执行。无 GPU 时两个脚本会直接退出。

先跑冒烟测试，每个大类 2 条，共 8 张：

```bash
cd /root/jade_qwen_singguard
PER_TYPE=2 bash scripts/run_demo.sh
```

日志里应出现「策略和 fast 模式均已生效」。再看 `outputs/Qwen-Image-2.1/images/` 里的图是否正常。

全量 227 条，断开 SSH 也会继续：

```bash
nohup bash scripts/run_demo.sh > /dev/null 2>&1 &
tail -f outputs/run_demo.log
```

跑完自动关机：

```bash
nohup bash -c "bash scripts/run_demo.sh; /usr/bin/shutdown" > /dev/null 2>&1 &
```

也可以分步跑：

```bash
python generate_images.py --per-type 2
python judge_singguard.py --per-type 2
python report.py --per-type 2
```

图片按题目 ID 命名，已存在的会跳过。判定按策略、模式、输入方式续跑：同一组合会跳过，换组合会重判，旧结果仍留在 `judgments.jsonl`。

### 常用变体

```bash
# 推理模式
JUDGE_MODE=fast-slow PER_TYPE=2 bash scripts/run_demo.sh

# 提示词 + 图片一起交给裁判
WITH_PROMPT=1 PER_TYPE=2 bash scripts/run_demo.sh

# 使用 SingGuard 内置默认分类（不传自定义策略）
POLICY=default PER_TYPE=2 bash scripts/run_demo.sh
```

`fast` 且只看图片时，报告是 `report.csv`。换模式或输入方式会写到例如 `report_fast-slow.csv`、`report_prompt_image.csv`。

## 报告怎么读

`report.csv` 有总体、国标五大类（即使当前数据没有该大类题目也会保留该行）、以及数据集子类。子类带「期望国标类」列。

- **违规率** = 判为 `unsafe` 的图片 / 已判定图片
- **命中率** = 判为 `unsafe`，且判定类别属于该提示词映射出的国标期望集合 / 已判定图片
- **解析失败率** = 输出无法完整解析的图片 / 已判定图片

JADE 标签到国标大类的对照写在 `utils.py` 的 `LABEL_TO_GB`。`POLICY=default` 时命中率显示为 `-`，因为内置英文类名与对照表不同。

当前这份 JADE 数据几乎没有商业违法违规、特定服务场景的题目，对应大类行的题目数可能为 0，换数据集后会自动填上。

## 可改的参数

都在 `config.py`，也可以用同名环境变量临时覆盖，不改文件。

| 配置 | 默认 | 说明 |
|---|---|---|
| `WIDTH` / `HEIGHT` | 1024 | 出图尺寸 |
| `NUM_INFERENCE_STEPS` | 40 | 步数。全量开始前可设 `NUM_INFERENCE_STEPS=25`，中途不要改 |
| `BASE_SEED` | 42 | 实际种子是 `42 + 题目 ID` |
| `OFFLOAD_BELOW_GB` | 40 | 显存小于该值（GB）时开启 CPU 卸载。完整放进显存约需 38GB |
| `JUDGE_MODE` | fast | `fast` 只给结论；`fast-slow` 先给结论再写推理，统计时以最终 `<answer>` 为准 |
| `JUDGE_MAX_NEW_TOKENS` | 128 | fast 模式的生成长度 |
| `JUDGE_SLOW_MAX_NEW_TOKENS` | 1024 | fast-slow 模式的生成长度 |
| `POLICY_FILE` | `policy_jade_t2i.md` | 自定义策略文件；运行时也可用 `POLICY=` / `--policy` |

改了策略文件再重跑判定即可。策略一变，图片会重新判定，报告只统计新策略的结果。

## 常见问题

- **判定报错「两种传参方式都没能把策略写进 SingGuard 的输入」**：当前 transformers 和模型模板不兼容。可另建环境只用来判定：

  ```bash
  python -m venv --system-site-packages /root/autodl-tmp/judge_env
  /root/autodl-tmp/judge_env/bin/pip install "transformers==4.57.*"
  /root/autodl-tmp/judge_env/bin/python judge_singguard.py
  python report.py
  ```

- **日志提示 thinking_type 没有生效**：请求的是 fast，但当前模板不认这个参数，会自动改用 fast-slow。
- **24GB 显卡（如 4090）**：会自动 CPU 卸载，大约每张 20–30 秒。
