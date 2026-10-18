# JADE + Qwen-Image-2.1 + SingGuard 文生图安全评测

用 [JADE-T2I 中文提示词](https://github.com/whitzard-ai/jade-db/tree/main/jade-t2i-v1.0)依次完成：

1. Qwen-Image-2.1 生成图片。
2. SingGuard-8B 按 `policy_jade_t2i.md` 的国标五大类，或指定策略判断图片。
3. 汇总覆盖率、违规率、类别命中率和解析失败率，输出 CSV。

实际样本数以本地 CSV 为准；上游版本和筛选方式会影响条数。此项目针对 Linux / AutoDL GPU 实例运行完整推理；Windows 本机可检查数据、任务计划和运行离线测试。数据集和生成图片可能包含违规内容，`data/`、`outputs/` 已在仓库根目录 `.gitignore` 中忽略。

## 文件与输出

```text
jade_qwen_singguard/
├── config.py              # 数据、模型、输出路径与推理参数
├── utils.py               # CSV/JSONL、缓存匹配、策略解析、标签映射
├── generate_images.py     # 出图
├── judge_singguard.py     # 判定
├── report.py              # 报告
├── policy_jade_t2i.md     # 国标五大类及小类说明
├── requirements.txt
├── tests/                 # 不加载模型的回归测试
├── data/jade_benchmark_zh.csv  # 需另行准备，未随仓库提交
└── scripts/run_demo.sh    # Linux Bash 顺序入口
```

默认输出为 `outputs/Qwen-Image-2.1/`：

| 文件 | 内容 |
|---|---|
| `images/{id}.png` | 生成图片 |
| `generations.jsonl` | 出图输入、参数指纹、图片 SHA-256、耗时或异常 |
| `judgments.jsonl` | 裁判原文、解析结果、数据/图片/模型/策略等匹配信息 |
| `report_{策略指纹}.csv` | 国标策略或其他文件策略 + fast + 只看图片的报告 |
| `report_default.csv` | SingGuard 内置策略 + fast + 只看图片的报告 |
| `report_{策略指纹}_{模式或输入方式}.csv` | 同一策略其他模式、输入方式的报告 |

所有正式报告均包含策略指纹，例如 `report_a1b2c3d4.csv`、`report_a1b2c3d4_fast-slow.csv`、`report_a1b2c3d4_prompt_image.csv`；内置策略指纹为 `default`。修改仓库自带策略也会产生新文件名，不再写无指纹的 `report.csv`。重复导出同一组合会更新该 CSV；冒烟测试与全量实验建议用不同 `OUTPUT_DIR`，以保留各自报告。入口日志默认为 `outputs/run_demo.log`，可用 `LOG_FILE` 改路径。

## 准备环境

完整推理需要 Python ≥ 3.10、匹配 CUDA 的 PyTorch ≥ 2.6 和 NVIDIA GPU。先确认 AutoDL 镜像里的 torch/CUDA，再安装其余依赖：

```bash
cd /root/LLM_Quick_Use/text_to_image_generate_safety_evaluate/jade_qwen_singguard
python -c "import sys, torch; print(sys.version); print(torch.__version__, torch.version.cuda); print('GPU:', torch.cuda.is_available())"
python -m pip install -r requirements.txt
python -m pip check
python -c "from diffusers import QwenImage21Pipeline; from transformers import AutoModelForImageTextToText; print('imports ok')"
```

`diffusers>=0.41.0` 已包含 `QwenImage21Pipeline`，无需追踪未固定版本的 GitHub 主分支。见 [Diffusers 0.41.0 发布](https://pypi.org/project/diffusers/0.41.0/)与[该版本源码](https://github.com/huggingface/diffusers/blob/v0.41.0/src/diffusers/__init__.py)。依赖范围用于安装；正式实验通过冒烟测试后保存 `python -m pip freeze > outputs/requirements.resolved.txt`，并记录 Python、torch、CUDA 和 GPU 型号。

## 准备数据和模型

在本目录下载 CSV，或用环境变量 `DATA_CSV` 指向已有文件：

```bash
mkdir -p data
curl -fL --retry 3 \
  https://raw.githubusercontent.com/whitzard-ai/jade-db/main/jade-t2i-v1.0/jade_benchmark_zh.csv \
  -o data/jade_benchmark_zh.csv
sha256sum data/jade_benchmark_zh.csv > data/jade_benchmark_zh.csv.sha256
```

CSV 必须包含 `ID`、`提示词`、`违规类型`、`违规子类` 表头。ID 必须是非负整数且唯一，提示词不可为空；空文件和格式问题会直接报错。保留数据 SHA-256 和下载时的上游提交版本，复现实验时使用同一 CSV。

模型文件需在正式运行前单独下载到本地。参考官方模型页 [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) 和 [SingGuard-8b](https://huggingface.co/inclusionAI/SingGuard-8b)，保存完整权重、配置、处理器与模板文件。默认读取：

```text
/root/autodl-tmp/models/Qwen-Image-2.1   # 根目录含 model_index.json
/root/autodl-tmp/models/Sing-Guard-8b   # 根目录含 config.json
```

ModelScope 风格的 `snapshots/master` 子目录会自动识别。也可以显式设置 `T2I_MODEL_DIR` / `JUDGE_MODEL_DIR`。两个推理脚本仅加载本地文件；不会在缺少权重时自动下载。磁盘和显存需求以模型实际文件、精度与卸载方式为准。记录模型仓库 revision；同一路径替换权重后请使用新的 `OUTPUT_DIR`，缓存指纹不逐个散列模型权重。

## 本机检查（无需模型和 GPU）

只检查 CSV、策略、参数及当前任务计划：

```bash
python generate_images.py --per-type 2 --dry-run
python judge_singguard.py --per-type 2 --dry-run
```

两步都不导入 torch、diffusers 或 transformers，不加载权重，不生成图片，不写入判定结果；读取已有图片的校验需要 Pillow。缺图会提示待生成，首次运行也可检查计划。`--dry-run` 通过只表示这些离线检查通过，不证明权重、CUDA、模型模板或真实推理可运行。

运行离线回归测试（在本目录执行）：

```bash
python -m pip install pillow
python -m unittest discover -s tests -v
```

Linux / Bash 也可以使用：

```bash
DRY_RUN=1 PER_TYPE=2 bash scripts/run_demo.sh
```

`DRY_RUN=1` 跳过报告，避免生成看似完成的评测结果。Windows PowerShell 请使用上述 Python 命令；`run_demo.sh` 需要 Bash 环境。

## GPU 冒烟测试与全量运行

先用每个数据集首个大类标签的前 2 条测试；多标签样本只按第一个标签抽样，最终条数以日志为准：

```bash
PER_TYPE=2 bash scripts/run_demo.sh
```

也可以分步运行：

```bash
python generate_images.py --per-type 2
python judge_singguard.py --per-type 2
python report.py --per-type 2
```

检查图片、裁判原始输出与报告覆盖率。策略和指定模式不能进入模型模板时直接报错，不会把 fast 静默改为 fast-slow；如需更深入推理，应明确选择 `--mode fast-slow`。

全量运行可保留到独立目录；断开 SSH 后继续：

```bash
OUTPUT_DIR=outputs/Qwen-Image-2.1-full nohup bash scripts/run_demo.sh > /dev/null 2>&1 &
tail -f outputs/run_demo.log
```

希望仅在任务成功后关机，可执行：

```bash
nohup bash -c 'bash scripts/run_demo.sh && /usr/bin/shutdown -h now' > /dev/null 2>&1 &
```

未能修复的生成失败、判定阶段缺图或图片损坏、判定异常或解析失败都会导致对应脚本非零退出，入口随即停止。日志和已有成功结果保留；处理问题后用同一配置重跑。报告可独立导出当前进度，未完成或解析失败会给出警告。

### 常用变体

```bash
# fast-slow 推理
JUDGE_MODE=fast-slow PER_TYPE=2 bash scripts/run_demo.sh

# 提示词作为 user、图片作为 assistant 一起给裁判
WITH_PROMPT=1 PER_TYPE=2 bash scripts/run_demo.sh

# SingGuard 内置分类
POLICY=default PER_TYPE=2 bash scripts/run_demo.sh

# 其他策略文件，路径中有空格时保留引号
POLICY='/path/to/custom policy.md' PER_TYPE=2 bash scripts/run_demo.sh

# 指定 Python 环境
PYTHON_BIN=/root/autodl-tmp/judge_env/bin/python PER_TYPE=2 bash scripts/run_demo.sh
```

同一入口的三个步骤使用 `PYTHON_BIN` 指定的同一解释器。如果出图和判定使用不同环境，使用分步命令，分别指定各自 Python；不要在入口执行到一半时改变环境。

## 断点续跑与实验隔离

- 出图只跳过匹配当前题目、提示词、模型目录、尺寸、步数、种子指纹，且内容 SHA-256 匹配的有效 PNG。仅仅存在同名 PNG 不代表可复用。
- 旧版没有元数据/指纹的图片，或同名 ID 对应提示词、配置已经变化时，会要求使用新 `OUTPUT_DIR`，或先备份再移走冲突图片。损坏图片会重新生成。
- 判定和报告先校验图片对应的生成元数据与当前出图配置、图片 SHA-256 匹配；分步运行裁判也必须保留对应的 `generations.jsonl`，不能只复制同名图片。再按数据内容、裁判模型目录、策略指纹、实际模式和输入方式匹配已有判定；解析失败和运行异常会重试。旧版缺少匹配信息的判定不复用，原日志仍保留。
- 策略变化只需重跑判定和报告。更换模型权重、推理环境或出图配置，建议新建 `OUTPUT_DIR`，保留独立实验结果。请勿同时启动多个进程写同一输出目录。

## 报告口径

表中同时给出 `题目数`、`已判定`、`有效判定`、`可对照数`，便于检查统计分母：

- **已判定**：当前实验匹配到裁判原始输出的样本数，包含解析失败；运行异常和缺图不计入。
- **有效判定**：已完整解析为 safe/unsafe 及合法类别的样本数。
- **违规率**：有效判定中 `unsafe` 数 / 有效判定数。
- **可对照数**：有效判定中，JADE 标签能够映射到国标期望类别的样本数。
- **命中率**：判为 `unsafe` 且类别落在期望集合的样本数 / 可对照数。
- **解析失败率**：未能完整解析的输出数 / 已判定数。

分母为 0 时显示 `-`。命中率衡量数据集提示词类别与生成图片判定类别的一致性，不是基于人工图片标注的准确率。应先看完成覆盖率，再读这些比例，不能把未判定或解析失败视为安全。

使用完整国标五大类策略时，保留五个大类行（无样本也保留），再展示数据集子类；`utils.py` 的 `LABEL_TO_GB` 维护 JADE→国标映射。SingGuard 内置分类、或名称不匹配完整国标五大类的其他策略，按原始 JADE 大类分组，命中率和可对照数显示为 `-`。多标签题目可能进入多个分组，分组题目数不应相加作为总体。

## 参数

环境变量覆盖 `config.py` 的输入配置；派生路径如 `IMAGE_DIR` 随 `OUTPUT_DIR` 计算，不能单独覆盖。

| 变量 | 默认 | 说明 |
|---|---|---|
| `DATA_CSV` | `data/jade_benchmark_zh.csv` | 数据文件 |
| `MODEL_ROOT` | `/root/autodl-tmp/models` | 默认模型根目录 |
| `T2I_MODEL_DIR` / `JUDGE_MODEL_DIR` | 上述模型目录 | 显式指定完整本地模型路径 |
| `T2I_MODEL_NAME` | 出图模型目录名 | 默认输出目录名；只允许单个目录名 |
| `OUTPUT_DIR` | `outputs/{T2I_MODEL_NAME}` | 每个实验建议独立目录 |
| `WIDTH` / `HEIGHT` | 1024 | 32 的正整数倍；模型兼容性仍需冒烟验证 |
| `NUM_INFERENCE_STEPS` | 40 | 正整数步数 |
| `BASE_SEED` | 42 | 每张图片的种子为 `BASE_SEED + ID` |
| `OFFLOAD_BELOW_GB` | 40 | 显存小于阈值时开启 CPU 卸载 |
| `JUDGE_MODE` | fast | `fast` / `fast-slow` |
| `JUDGE_MAX_NEW_TOKENS` | 128 | fast 输出上限 |
| `JUDGE_SLOW_MAX_NEW_TOKENS` | 1024 | fast-slow 输出上限 |
| `POLICY_FILE` | `policy_jade_t2i.md` | 配置默认策略；CLI `--policy` / 入口 `POLICY` 可覆盖 |

种子有助于同环境重跑，但不同 GPU、算子和库版本不保证图片逐字节一致。模型模板兼容、显存峰值、真实出图质量和裁判效果，仍需在目标 GPU 上完成冒烟测试后确认。
