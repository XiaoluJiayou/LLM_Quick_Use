# 数据抽样与问题生成

两个抽样脚本只依赖 Python 标准库，均支持 `--input`、`--output`、`--count`、`--seed`。随机抽样使用 reservoir 算法；分层抽样先统计类别，再逐类 reservoir 抽样，内存随样本量和类别数增长，不需要把整个 CSV 加载进内存。

```powershell
python data_generate/data_random_sample.py --input input.csv --output sampled.csv --count 1000 --seed 42
python data_generate/large-scale_data_random_sample.py --input input.csv --output stratified.csv --count 100 --category-column 一级领域
python csv_file_merge.py --input-dir csv_files --output merged.csv
```

抽样数量超过输入时返回全部数据，负数报错，数量为零输出表头。分层算法包含空分类，按比例分配后补齐整数余数，保证最终行数准确。同一输入和随机种子得到相同结果。CSV 表头和列数必须有效。

输出默认不能覆盖已有文件，需要覆盖时显式加 `--overwrite`。所有 CSV 先写同目录临时文件，再原子替换；写出失败保留旧文件。合并会排除同路径的旧输出，按文件名排序；列名不同时默认报错，确需取并集时加 `--allow-schema-union`。

## 两个生成 Notebook

在项目根目录安装公共模块和数据依赖：

```powershell
python -m pip install -e ".[data]"
```

另按运行环境安装 `modelscope`、`torch`、`transformers`、`accelerate`、`jupyter`；模型推理版本应与服务器 CUDA 配置匹配。启动 Notebook 前配置：

| 环境变量 | 用途 |
| --- | --- |
| `LLM_MODEL_PATH` | 必填，已下载的本地模型目录 |
| `DATASET_INPUT` | 种子 Excel 路径，默认 `政治专项数据集.xlsx`，需要 `Prompt` 列 |
| `DATASET_OUTPUT` | 问题 JSON 路径，默认 `questions.json` |
| `SEED_LIMIT` | 批量 Notebook 最多处理的种子数，默认 100 |

模型输出必须为 JSON 字符串数组。解析通过公共 JSON 解码器支持字符串中的括号；无效结果会明确报错或记录当次失败，不再静默变为空结果。问题写入时去重并原子替换，已有 JSON 损坏时保留文件并停止。Notebook 采用单写入进程；不要让两个任务同时写同一个结果文件。

离线测试验证代码单元语法、解析、数据保护和 CSV 工作流；GPU 内存、模型加载和生成质量仍需在配置了权重的机器上验证。
