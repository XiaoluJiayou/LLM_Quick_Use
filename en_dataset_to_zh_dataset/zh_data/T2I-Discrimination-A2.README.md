# T2I-Discrimination-A2 中文歧视风险提示词

构建日期：2026-10-19。独立正式文件为 [`T2I-Discrimination-A2.zh.json`](T2I-Discrimination-A2.zh.json)，共 **2,000 条全中文文生图提示词**。现按用户要求全部追加到 [`T2I-RiskyPrompt.zh.json`](T2I-RiskyPrompt.zh.json)，合并后共 8,261 条；独立 A2 集仍保留。A5 未纳入本次工作。

新增 ID 为 `6433` 至 `8432`，通过 [`T2I-RiskyPrompt.zh.a2-merge.json`](T2I-RiskyPrompt.zh.a2-merge.json) 映射回本集 ID `0` 至 `1999`，再查询逐条溯源。原 6,261 条记录保持不变，合并前文件保存在 [`T2I-RiskyPrompt.zh.before-A2.json`](T2I-RiskyPrompt.zh.before-A2.json)。

## 文件和格式

| 文件 | 用途 |
| --- | --- |
| [`T2I-Discrimination-A2.zh.json`](T2I-Discrimination-A2.zh.json) | 正式数据；与参考集一样，仅有 `prompt`、`label` 两个字段 |
| [`T2I-Discrimination-A2.provenance.json`](T2I-Discrimination-A2.provenance.json) | 与正式数据 ID 一一对应的原文、来源行号、固定版本、中文改写及复核记录 |
| [`T2I-Discrimination-A2.excluded.json`](T2I-Discrimination-A2.excluded.json) | 3,797 条工作候选中未进入正式集的 1,797 条及原因，包括已复核但未选用的余量 |
| [`T2I-Discrimination-A2.validation.json`](T2I-Discrimination-A2.validation.json) | 可离线复跑的校验结果和成品 SHA-256 |

顶层是 JSON 对象，ID 为连续字符串 `"0"` 至 `"1999"`。每条结构为 `{"prompt": "中文绘图指令或画面描述", "label": ["包含歧视性内容"]}`。字段名及标签格式遵循参考集。

## 来源组成

| 来源 | 正式条数 | 中文处理及 T2I 形式 |
| --- | ---: | --- |
| CoProV2 / SafetyDPO | 485 | 英文原生画面提示词，逐条对照原文作中文翻译和校订 |
| DETONATE | 1,494 | 保留原文歧视对象和主要陈述，逐条转为中文，再适配为海报、标语牌、社交配图、墙面文字或卡片 |
| JADE-T2I v2 | 12 | 原生中文，原文不改 |
| JADE-T2I v1 | 9 | 原生中文，原文不改 |
| 合计 | **2,000** | **506 条画面描述、1,494 条中文文字入图任务** |

按“CoProV2 优先、DETONATE 补足、JADE 补本土语境”的顺序筛源；没有强行维持 CoProV2 的数量占比。CoProV2 的 unsafe Hate 原始候选有 4,393 行，但包含中性身份描述、歧义词、类别噪声和本次未采用的性内容等。严格筛选后的工作候选为 490 条，经过语境复核和去重，正式保留 485 条，因此最终数量主要由 DETONATE 补足。

DETONATE 的原始内容很多是社媒发言，不是现成画面描述。本次将其明确转成**中文文字入图任务**，没有把抽象辱骂冒充原生画面提示词。每条原文最多形成一条正式数据；五种载体只分配不同样本，不把同一原文换五种样式凑数。测评时建议通过 `rendering_format_added` 分开统计两种任务，因为文字识别、文字生成能力会影响文字入图结果。

## 国标映射和标签含义

依据本仓库的 [GB/T 45654-2025 PDF](../../text_to_image_generate_safety_evaluate/jade_qwen_singguard/GBT+45654-2025.pdf) 附录 A.2，以及[类别转录](../gbt_categories.md)。每条指定一个主要小类，用于索引和统计，允许实际内容跨小类或跨 A.1/A.4 等风险。

| 主要小类 | 条数 |
| --- | ---: |
| A.2.a 民族歧视；本项目包含种族相关歧视的工作映射 | 1,406 |
| A.2.b 信仰歧视 | 89 |
| A.2.c 国别歧视 | 53 |
| A.2.d 地域歧视 | 19 |
| A.2.e 性别歧视 | 53 |
| A.2.f 年龄歧视 | 49 |
| A.2.g 职业歧视 | 1 |
| A.2.h 健康歧视 | 52 |
| A.2.i 其他方面歧视 | 278 |

这是 **A2 专项风险内容集，并非九小类均衡集**。其他方面歧视包括性取向、性别表达、体型、社会经济身份等。`white trash` 一类阶层贬称归入其他方面，而非仅凭出现该词就归为地域歧视。泛指“蠢”的辱骂不会只因出现 `retard` 就算健康歧视；民族、信仰、国别等边界有交叉，`primary_subtype` 是本项目工作标注，不代表专家共识。

标签表示提示词包含歧视性称谓、刻板印象、排斥、贬损或歧视场景。JADE 等来源也包含**描绘他人受到歧视**的场景；这不等于提示词在赞同歧视。不能直接把全部条目当成“必须拒绝”的金标准，仍需结合生成意图和实际输出判断。本集只标 A2，**未穷尽其他风险标签**，所以单个 A2 标签不表示内容不涉及暴力、侵权等其他风险。

## 中文处理和复核

早期离线机翻出现俚语误译、漏译，因此未直接把机翻结果作为正式交付。最终所有 1,979 条英文来源均在本轮由助手逐条对照原文翻译、校订或作中文适配；21 条中文来源经本轮语境复核，保持原文。没有用英文原文入图替代中文。

处理保留主要歧视对象、贬损含义和场景，不通过增加新攻击对象或新事件扩充样本。个别社媒原文的非关键话语、重复笑声、无关口头语作了省略，故统一记录为 `source-based session Chinese translation/adaptation`，不宣称每条都是逐字直译。外语歧视性称谓采用可理解的中文对应表达；这不意味着来源事件发生在中国，或攻击对象被改成中国人。中文引号及“被蔑称为”等表述用于标记称谓性质。

复核、排除和术语修订记录保存在：

- [`a2_reviewed_translations.tsv`](../a2_reviewed_translations.tsv)：CoProV2 中文校订。
- [`a2_review_batches/`](../a2_review_batches/)：DETONATE 的逐条中文文本或排除理由，`!` 开头表示排除。
- [`a2_review_edits.json`](../a2_review_edits.json)：额外排除与主要小类修订。
- [`a2_review_bindings.json`](../a2_review_bindings.json)：上述记录与原始文件、行号、原文 SHA-256 的绑定；来源错位时构建会报错。

这是助手在本轮完成的语境与译文复核，**不是独立人工双标或专家审核**。去除了明确反歧视转述、仅提及身份的中性内容、无关个人争执、难以理解的断句/影射、未恢复的占位符、部分其他风险过强的候选等。筛选未覆盖所有潜在语义错误，校验也不认证法律适用性。

## 溯源和许可

原始 CSV、下载清单与原仓库许可证已保存在 [`en_data/a2_sources/`](../en_data/a2_sources/)，清单包含逐文件 URL、字节数、SHA-256 和固定修订：

| 来源 | 作者发布入口 | 本次固定修订 | 原仓库许可 |
| --- | --- | --- | --- |
| CoProV2 | [Visualignment/SafetyDPO](https://github.com/Visualignment/SafetyDPO) | `5f71f8db885f096f84abd391f48e49e78b32e8d1` | MIT |
| DETONATE | [abhilekhborah/Detonate](https://github.com/abhilekhborah/Detonate) | `e6c3214948eef85e33cf5b8175c741dd79db161a` | Apache-2.0；保留 `upstream_dataset`、`upstream_source`，上游文本许可不能由该许可证一并推定 |
| JADE v1 | [whitzard-ai/jade-db](https://github.com/whitzard-ai/jade-db) | `e8b949936b08e94434f54f8c20de6c1516122257` | MIT；项目另有学术研究用途声明 |
| JADE v2 | [whitzard-ai/jade-db](https://github.com/whitzard-ai/jade-db) | `cfb889959e15d49e79b00e3e7dc52e0e232e1ca8` | 同上 |

本汇编用于安全研究和评测，不对上游内容重新授予统一商用许可。未抓取或生成图像。来源含公开训练/评测材料；本次 2,000 条不构成与上游训练材料隔离的全新测试集。

`provenance` 中 `source_row_zero_based` 是 CSV 去掉表头后的零基数据行号；`source_id` 保留来源自己的 ID。`candidate_a2_matches` 记录候选规则命中，**不能当成全部已确认的小类标注**。正式工作小类看 `primary_subtype`，最终中文文本看 `zh_translation` 和 `final_prompt`。

## 校验与复现

从仓库根目录运行，仅依赖 Python 标准库，不需要模型、推理服务、密钥或联网：

```powershell
python en_dataset_to_zh_dataset/validate_a2_dataset.py --write-report
```

完整重建使用已保存的原始 CSV 和校订表：

```powershell
./en_dataset_to_zh_dataset/rebuild_a2.ps1
```

只有需要重新下载固定版本时，才增加 `-DownloadSources`。重建会改写本补充集与其报告，不会修改参考集。`a2_work/` 是可重建的中间目录，已加入忽略规则；正式交付不依赖其中的机翻缓存或模型权重。

检查覆盖：2,000 条及字段结构、连续 ID、中文文本、无英文长词回退、无未处理标记、正式文本/入图前文本/原文的归一化完全去重、字符三元组 Jaccard 不低于 0.90 的高相似文本检查、来源文件哈希和行号复原、中文来源不改写、逐条复核状态及译文校验和、合并前参考备份不变、与该备份无归一化完全重合。构建及校验会自动使用合并前备份，避免已追加的 A2 被误当成原始重复项。高相似文本检查不等于语义去重；模板和相近主题仍然存在。

参考集原始 SHA-256：`4fd4a7f38577c8d3aa146ef8da33721337ff26a7c36f2758d8db828f9af9eee4`。正式成品 SHA-256 以本目录的 `validation.json` 为准。
