# 英文数据集转中文并映射国标五大类

类别映射依据：**GB/T 45654-2025《网络安全技术 生成式人工智能服务安全基本要求》附录 A**（资料性）「训练数据及生成内容的主要安全风险」。输出时归入下列五大类之一（共 31 项小类，原文见 [gbt_categories.md](gbt_categories.md)，与标准 PDF 附录 A 一致）：

1. **违反社会主义核心价值观**（A.1，8 项）：煽动颠覆国家政权、推翻社会主义制度；危害国家安全和利益、损害国家形象；煽动分裂国家、破坏国家统一和社会稳定；宣扬恐怖主义、极端主义；宣扬民族仇恨；宣扬暴力、淫秽色情；传播虚假有害信息；其他法律、行政法规禁止的内容。
2. **歧视性内容**（A.2，9 项）：民族、信仰、国别、地域、性别、年龄、职业、健康、其他方面歧视内容。
3. **商业违法违规**（A.3，5 项）：侵犯他人知识产权；违反商业道德；泄露他人商业秘密；利用算法、数据、平台等优势实施垄断和不正当竞争；其他商业违法违规行为。
4. **侵犯他人合法权益**（A.4，7 项）：危害他人身心健康；侵害他人肖像权、名誉权、荣誉权、隐私权、个人信息权益；侵犯他人其他合法权益。
5. **无法满足特定服务类型的安全需求**（A.5，2 项）：面向关键信息基础设施、自动控制、医疗信息服务、心理咨询、金融信息服务等场景的内容不准确或内容不可靠。

无法归入任一类时，输出 `无法映射`。完整小类条文以 [gbt_categories.md](gbt_categories.md) 为准（该文件会写入模型提示词）。标准原文 PDF 见评测目录中的 `GBT+45654-2025.pdf`。

## 安装

```bash
# 在项目根目录执行（注册公共工具模块）
python -m pip install -e ".[data,api]"
```

## 配置模型（远程 API 或本地部署）

脚本使用 OpenAI 兼容的 Chat Completions，远程与本地共用同一套代码。

| 环境变量 | 说明 | 示例 |
| --- | --- | --- |
| `OPENAI_BASE_URL` | 接口根路径 | `https://api.openai.com/v1` 或 `http://127.0.0.1:8000/v1` |
| `OPENAI_API_KEY` | API Key；本地可填任意非空字符串 | `sk-...` |
| `OPENAI_MODEL` | 模型名 | `gpt-4o-mini` / `qwen2.5-14b` |

PowerShell 示例：

```powershell
$env:OPENAI_BASE_URL = "http://127.0.0.1:8000/v1"
$env:OPENAI_API_KEY = "local"
$env:OPENAI_MODEL = "qwen2.5-14b"
```

本地可用 vLLM、SGLang、Ollama 等开启兼容 `/v1/chat/completions` 的服务，再把 `OPENAI_BASE_URL` 指过去。

## 目录

| 目录 | 含义 |
| --- | --- |
| `en_data/` | 英文原始数据 |
| `zh_data/` | 转换后的中文数据 |
| `configs/` | 各数据集的 yaml 配置 |

## 数据集配置

每种数据集一份 yaml，放在 `configs/`（见 [config.example.yaml](config.example.yaml)）：

| 字段 | 含义 |
| --- | --- |
| `input` | 输入文件路径（通常在 `en_data/`） |
| `format` | `csv` / `json` / `jsonl` |
| `json_key` | 可选；json 列表包在某个键下时填写 |
| `id_field` | 可选；断点续跑用的唯一 id，缺省用行号 |
| `text_field` | 要翻译的英文指令列 |
| `category_field` | 原类别列（字符串或列表均可） |
| `output` | 输出路径（通常在 `zh_data/`），格式与输入相同 |

只翻译 `text_field`，其他列原样保留，并追加：

- `中文指令`
- `国标类别`
- `原类别`

## 运行

```bash
# 在项目根目录先检查输入、配置及续跑来源；不调用模型
python -m en_dataset_to_zh_dataset.convert --dataset configs/example.yaml --limit 3 --dry-run

# 小样核对，输出独立的 example_zh.limit-3.csv
python -m en_dataset_to_zh_dataset.convert --dataset configs/example.yaml --limit 3

# 全量
python -m en_dataset_to_zh_dataset.convert --dataset configs/example.yaml
```

进度写在输出文件旁的 `*.progress.jsonl`，中断后重跑会跳过已成功的 id。失败条目写入 `*.errors.jsonl`，不中断整批，命令以非零状态退出；同配置重跑会重试失败行。最终结果通过临时文件原子替换。

输入 ID 必须唯一且非空；没有 `id_field` 时使用行号。续跑文件记录完整输入、配置、分类说明、提示词、模型和服务地址的指纹；其中任何内容改变时会拒绝混用旧结果，需要新输出路径。令牌不写入指纹或结果。旧版缺少指纹的进度不会自动复用，损坏 JSONL 也不会静默忽略。

`--limit` 必须为正整数，输出名会自动添加 `.limit-N`，不覆盖全量结果。`--dry-run` 不需要模型凭据或 OpenAI SDK，但需要 PyYAML 读取配置；不会创建结果文件。字段缺失、空文本、重复 ID 和输入输出同路径都会在请求前报错。模型输出截断、错误 JSON 或未知类别不视为成功。
