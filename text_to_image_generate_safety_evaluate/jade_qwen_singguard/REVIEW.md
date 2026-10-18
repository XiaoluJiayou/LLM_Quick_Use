# 代码审查记录

审查日期：2026-10-18。基线提交：`801161a`。

## 结论与验证范围

已审查全部 5 个 Python 模块、Bash 入口、依赖文件、使用说明及策略与代码的衔接。模型调用结构已与官方示例核对。原实现存在续跑结果污染、统计偏差及失败未上报等问题，本次已修复。

修复后通过 **65 项离线回归测试**、全部 **10 个 Python 文件**的语法检查，以及 Bash 语法检查。完整流程测试使用真实临时 CSV、PNG、JSONL、CSV 报告，仅替换模型推理；能够确认结果文件衔接、统计结果与续跑行为，不能证明真实模型的推理效果、显存需求或吞吐。

本机默认环境为 Python 3.9.7、torch 1.9.1+cu111、transformers 4.6.1，缺少 diffusers；配置指向的 CSV、出图模型和裁判模型均不存在。本次使用独立的 Python 3.12.14 运行离线测试，未更改用户默认 Python 环境，也未下载模型。

## 已修复的问题

| 优先级 | 原实现的问题 | 修复后的行为 |
|---|---|---|
| P1 | 出图只根据 ID 对应 PNG 是否存在决定跳过；修改提示词、参数或模型后混用旧图 | 核验生成指纹、图片完整性、SHA-256；不匹配时要求新输出目录，损坏图片重新生成 |
| P1 | 裁判只按 ID、策略、模式和输入方式续跑，解析失败也被永久跳过 | 同时校验数据内容、生成来源、当前图片、裁判模型；只有有效判定跳过，失败可重试 |
| P1 | 分步运行裁判或报告时，旧图可能配上新提示词或新出图参数 | 两步也独立校验 `generations.jsonl`，旧图不能作为新实验的有效输入 |
| P1 | 解析失败进入比例分母，甚至用开头的 quick verdict 计为违规；最后答案截断可能误用前一个答案 | 只接受完整的最终答案，类别字母与名称必须一致；解析失败单独统计 |
| P1 | 默认英文策略用中文国标类别筛选分组，导致所有大类题目数为零 | 非完整国标策略按原始 JADE 标签分组，关闭无法对照的命中率 |
| P1 | 出图异常、缺图、裁判异常或解析失败后脚本仍返回成功 | 对应脚本返回非零，Bash 入口停止并保留成功结果 |
| P2 | fast 模式未生效就静默降级，实际模式与报告筛选条件不一致 | 尝试两种官方传参方式，验证完整策略及实际模式；不兼容则明确失败 |
| P2 | JSONL 最后一行残缺时直接追加，下一条成功记录也会被拼坏；截断 UTF-8 会导致读取失败 | 追加前补换行，逐行处理残缺 JSON、UTF-8 和非对象记录 |
| P2 | PNG 发布与元数据落盘之间中断，留下无法追溯来源的完整图片 | 临时 PNG → 记录指纹和哈希 → 发布 PNG；发布前中断可自动重生成 |
| P2 | CSV 重复 ID 会覆盖图片，`--per-type 0` 被当成全量 | 提前校验 CSV 表头、列数、ID 唯一性、提示词与种子范围；抽样参数必须为正整数 |
| P2 | 报告不区分策略，修改策略后覆盖旧文件 | 所有正式报告包含策略指纹，并按模式、输入方式区分文件名 |
| P2 | 依赖文件没有安装 diffusers，说明中的 Python/torch 版本要求偏低 | 补充可安装的依赖范围、数据准备步骤和目标 GPU 环境要求 |

此外，两个推理入口都支持无模型的 `--dry-run`，不再在查看帮助或检查数据时导入大型推理依赖；不支持 BF16 的 GPU 会使用 FP16。

## 报告口径

报告显式展示 `题目数`、`已判定`、`有效判定`、`可对照数`。违规率以有效判定为分母，命中率以有期望类别映射的有效判定为分母；解析失败率以已判定样本为分母。零分母显示 `-`，未完成报告会给出提示。

命中率反映提示词标签与生成图片判定的一致性，不能替代人工图像标注的准确率。策略中的图像化规则保留原文，未进行法律解释或法律合规认证；授权、真实性及具体服务场景等信息也不能仅由一张图确定。

## 使用与迁移

- 具体安装、数据准备和运行步骤见 [README.md](README.md)。
- 原版图片或判定若缺少新元数据，不会静默复用；建议使用新 `OUTPUT_DIR` 重新评测，原文件无需删除。
- 正式报告改为 `report_{策略指纹}[_模式][_输入方式].csv`，不再固定写 `report.csv`。
- 不逐个散列大型模型权重；同一路径替换权重或更换推理环境时，应新建输出目录并记录模型 revision 与依赖版本。
- 请勿让多个进程同时写同一输出目录。

在目标 Linux / AutoDL 环境按 README 安装依赖并准备数据、模型后，先验证：

```bash
python -m unittest discover -s tests -v
OUTPUT_DIR=outputs/review-smoke DRY_RUN=1 PER_TYPE=2 bash scripts/run_demo.sh
OUTPUT_DIR=outputs/review-smoke PER_TYPE=2 bash scripts/run_demo.sh
```

最后一条会执行真实 GPU 推理。需要检查生成图片、原始裁判输出、是否成功加载指定策略与模式、报告覆盖率，以及显存和耗时；本次尚未执行该步骤。

## 官方接口与依赖依据

- [Qwen-Image 2.1 模型卡](https://huggingface.co/Qwen/Qwen-Image-2.1)
- [Diffusers 0.41.0 的 QwenImage21Pipeline 文档](https://huggingface.co/docs/diffusers/v0.41.0/api/pipelines/qwenimage21)
- [Diffusers 0.41.0 依赖声明](https://github.com/huggingface/diffusers/blob/v0.41.0/setup.py)
- [Transformers 5.17.0 发布信息](https://pypi.org/project/transformers/5.17.0/)
- [SingGuard 官方模型卡](https://huggingface.co/inclusionAI/SingGuard-8b)
- [SingGuard 官方聊天模板](https://huggingface.co/inclusionAI/SingGuard-8b/blob/main/chat_template.jinja)
- [JADE-T2I 官方数据目录](https://github.com/whitzard-ai/jade-db/tree/main/jade-t2i-v1.0)
