# Qwen-Image-2.1 + 裁判模型文生图安全评测

用 `T2I-RiskyPrompt.zh.json`（国标 A.1–A.4）出图 → 裁判模型按 `policy_gbt_t2i.md` 判定 → 输出 CSV。面向 AutoDL GPU。

---

## 最快跑通（AutoDL）

本目录可单独打包上传（只需 `t2i_qwen_judge/`，不必带整个仓库）。解压后进入该目录即可。

### 1. 环境

镜像自带 CUDA + PyTorch ≥ 2.6，再装依赖：

```bash
cd /root/t2i_qwen_judge   # 按你实际上传/解压后的路径改
python -m pip install -r requirements.txt
```

### 2. 数据

上传包里请带上 `data/T2I-RiskyPrompt.zh.json`。若还没有：

```bash
mkdir -p data
# 把 T2I-RiskyPrompt.zh.json 放到 data/ 下即可
ls data/T2I-RiskyPrompt.zh.json
```

### 3. 模型（ModelScope，下到数据盘）

```bash
python download_models.py
```

会依次下载到：

- `/root/autodl-tmp/models/Qwen-Image-2.1`
- `/root/autodl-tmp/models/t2i-judge`

与 `config.py` 默认路径一致，一般无需再改 `T2I_MODEL_DIR` / `JUDGE_MODEL_DIR`。若服务器上仍是旧目录名，可显式指定：`export JUDGE_MODEL_DIR=/旧路径`。

### 4. 小样本试跑（每大类 100 条，约 400 张）

```bash
# 可选：不加载模型，只检查数据和任务
DRY_RUN=1 PER_TYPE=100 bash scripts/run_demo.sh

# 真跑（默认 JUDGE_MODE=fast-slow；只要结论时再加 JUDGE_MODE=fast）
OUTPUT_DIR=outputs/sample-per100 PER_TYPE=100 bash scripts/run_demo.sh
```

### 5. 全量（约 8261 条，很慢）

```bash
OUTPUT_DIR=outputs/Qwen-Image-2.1-full nohup bash scripts/run_demo.sh > /dev/null 2>&1 &
tail -f outputs/run_demo.log
```

不设 `PER_TYPE` 即全量。同配置重跑可续跑。

---

## 结果在哪

默认或你设的 `OUTPUT_DIR` 下：

| 文件 | 内容 |
|---|---|
| `images/{id}.png` | 生成图 |
| `generations.jsonl` | 出图记录 |
| `judgments.jsonl` | 裁判原文与解析 |
| `report_*.csv` | 汇总（违规率、命中率等） |
| `outputs/run_demo.log` | 入口日志 |

报告先看「已判定 / 题目数」是否够，再看违规率、命中率。命中率 = 判定类别是否落在数据标签期望集合，不是人工看图准确率。

---

## 常用开关

`JUDGE_MODE` 默认 `fast-slow`。两种模式判的是同一张图、同一份策略，差别只在裁判怎么写答案：

| 模式 | 输出 | 报告用哪一句 | 速度 |
|---|---|---|---|
| `fast` | 只给结论 | 这一句结论 | 更快，生成更短 |
| `fast-slow` | 先给一句快速结论，再写推理，最后再给答案 | 最后一个完整答案。开头那句和最终答案不一致时，以最终答案为准 | 更慢，每条最多约 1024 个 token |

换模式会另记一份判定，不会和另一种模式的结果混在一起。

```bash
JUDGE_MODE=fast PER_TYPE=100 bash scripts/run_demo.sh        # 只要结论
WITH_PROMPT=1 PER_TYPE=100 bash scripts/run_demo.sh          # 提示词+图一起判
POLICY=default PER_TYPE=100 bash scripts/run_demo.sh         # 裁判模型内置分类
```

上面的 `scripts/run_demo.sh` 会把下面三步一次跑完。想先出图、看完图再判定，就自己按这个顺序分开跑。后一步会读前一步写在同一个 `OUTPUT_DIR` 里的结果，顺序不能倒。`--per-type 100` 和试跑一样，每个大类只取 100 条；去掉它就是全量。判定和报告默认 `fast-slow`。个别图片没写出完整答案时，会用 2048 个 token 再判一次；仍然解析失败也继续汇总，不会挡住报告。出图失败、缺图或判定异常才会停。

```bash
python generate_images.py --per-type 100   # 1. 按提示词出图，保存到 images/
python judge.py --per-type 100             # 2. 用裁判模型判定这些图
python report.py --per-type 100            # 3. 把判定汇总成 report_*.csv
```

---

## 文件说明

| 文件 | 作用 |
|---|---|
| `config.py` | 路径与参数（可用环境变量覆盖） |
| `utils.py` | 读 JSON、续跑指纹、解析输出 |
| `generate_images.py` / `judge.py` / `report.py` | 出图 / 判定 / 报告 |
| `policy_gbt_t2i.md` | 国标五大类策略 |
| `scripts/run_demo.sh` | 三步串联 |
| `download_models.py` | ModelScope 下载两个模型 |

主要环境变量：`DATA_JSON`、`T2I_MODEL_DIR`、`JUDGE_MODEL_DIR`、`OUTPUT_DIR`、`JUDGE_MODE`、`PER_TYPE`（经脚本传入）。小样本试跑与全量请用不同 `OUTPUT_DIR`。
