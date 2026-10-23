# LLM Quick Use

面向模型实验、数据准备与内容安全评测的工具集合。保留各模型原有目录，公共工具提供配置化命令行、结构化结果、可恢复写出和离线验证。

本轮优化方案与实施记录见 [优化计划](docs/OPTIMIZATION_PLAN.md)，覆盖范围和实测结果见 [验收报告](docs/OPTIMIZATION_REPORT.md)。模型效果、云服务权限和 GPU 性能需在目标环境验收；离线测试使用模拟服务或随机小张量。

## 安装与环境

推荐 Python 3.10+。先创建独立环境，安装公共包；按任务选择依赖组：

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -e "."
.venv/Scripts/python -m pip install -e ".[data,test]"
```

Linux 激活方式为 `source .venv/bin/activate`，Windows PowerShell 可用 `.venv/Scripts/Activate.ps1`。下文 `python` 指已选择的项目环境。

| 依赖组 | 用途 |
| --- | --- |
| `data` | pandas、YAML、Excel、进度显示；CSV 合并与抽样本身只需标准库 |
| `api` | SDK 转换流程及 requests 云示例；公共 ChatClient 本身只用标准库 |
| `download` / `modelscope` | 分别安装 Hugging Face / ModelScope 下载 SDK |
| `clip` | ChineseCLIP 实验 |
| `train` | 教学 LDM、MNIST 与感知损失实验 |
| `video` | OpenCV 抽帧及媒体工具 |
| `qwen-vl` | Qwen3-VL 截图评估；需匹配权重与库版本 |
| `test` | 不下载模型的公共回归依赖；张量测试有 torch 时执行 |

例如 `python -m pip install -e ".[api,data]"`。GPU 环境先按设备 CUDA 版本安装匹配的 torch/torchvision，再安装任务依赖。Wan、LTX、Helios、LongCat、Janus、UniT2IXL 使用各自官方实现和独立环境；它们的专用代码不随本项目安装。Qwen-Image-2.1 / 裁判模型的版本要求见其[专门说明](text_to_image_generate_safety_evaluate/t2i_qwen_judge/README.md)。

`.env.example` 列出环境变量名，脚本读取当前进程环境；不会自动加载 `.env`。不要将凭据直接写进源码或命令行。生成文件建议放 `outputs/`、`results/`，下载权重放 `downloads/`，这些目录已忽略；原先已跟踪的样例媒体仍保留。

## 项目导航

| 模块 | 功能与入口 |
| --- | --- |
| [model_dataset_download](model_dataset_download/README.md) | 模型、数据集和单文件显式下载；支持无联网预览 |
| [data_generate](data_generate/README.md) | 随机/分层流式抽样、问题生成 Notebook |
| `csv_file_merge.py` | 流式合并 CSV，排除输出文件，检查列名 |
| [en_dataset_to_zh_dataset](en_dataset_to_zh_dataset/README.md) | 数据翻译和类别映射，支持指纹续跑和独立小样输出 |
| `model_response_generate` / `model_answer_safe_eval` | OpenAI 兼容聊天接口的批量回答与回答安全评估 |
| `batch_pretrain_data_processing_test.py` | 预训练语料的范围筛选与安全标注 |
| `OpenClip-Chinese-Project` | 单图、敏感词匹配及按批图片分类 |
| `base_ldm.py` / `full_ldm_unet.py` / `full_ldm_encode_decode.py` | 教学 LDM 与自编码器，支持无需数据的小张量检查 |
| [Text_and_Image_to_video_evaluate](Text_and_Image_to_video_evaluate/README.md) | 视频后端入口、全视频抽帧、截图评估与 Base64 工具 |
| [video-content-safety-evaluate](video-content-safety-evaluate/README.md) | 阿里云、百度视频审核的显式调用示例 |
| [T2I RiskyPrompt + Qwen + 裁判模型](text_to_image_generate_safety_evaluate/t2i_qwen_judge/README.md) | 带来源校验的文生图安全评测 |
| `text_to_image_generate_safety_evaluate/*.ipynb` | Janus、UniT2IXL、LlavaGuard 探索示例；按实际数据长度执行 |
| `main.py` / `leetcode.py` / `linked_list_intersection.py` / `video-content-safety-evaluate/prim_test.py` | 独立学习示例 |
| `llm_quick_use` / `tools` / `tests` | 公共工具、项目检查和离线回归 |

`text_to_image_generate_safety_evaluate/package.json` 的内容是历史 Docker daemon 配置示例，不是 Node.js 依赖清单，不用于安装前端项目。使用镜像配置前请自行核实来源与可用性。

## 数据处理示例

在项目根目录执行，先 `--help` 查看参数：

```powershell
python data_generate/data_random_sample.py --input source.csv --output outputs/sample.csv --count 100 --seed 42
python data_generate/large-scale_data_random_sample.py --input source.csv --output outputs/stratified.csv --count 100 --category-column 一级领域 --seed 42
python csv_file_merge.py --input-dir csv_parts --output outputs/merged.csv
python model_dataset_download/model_download_huggingface.py --repo-id openai-community/gpt2 --local-dir downloads/gpt2 --dry-run
```

抽样使用 reservoir，内存随样本量增长，不随完整 CSV 行数增长。分层抽样包含空分类并精确分配总量。现有输出需显式 `--overwrite`，输出不能覆盖输入；CSV 合并默认要求相同列名，可用 `--allow-schema-union` 明确取并集。

## 文本回答与安全评估

公共批处理的 `LLM_BASE_URL` / `--endpoint` 是**完整聊天补全端点**，例如自建服务的 `http://127.0.0.1:8000/v1/chat/completions`。`LLM_MODEL` 是部署模型名；鉴权来自 `LLM_API_KEY`。

```powershell
python model_response_generate/batch_model_response_generate.py --input prompts.csv --output outputs/answers.csv --endpoint http://127.0.0.1:8000/v1/chat/completions --model deployed-model --dry-run
python model_response_generate/batch_model_response_generate.py --input prompts.csv --output outputs/answers.csv --endpoint http://127.0.0.1:8000/v1/chat/completions --model deployed-model
python model_answer_safe_eval/batch_model_response_safe_eval.py --input outputs/answers.csv --output outputs/judged.csv --endpoint http://127.0.0.1:8000/v1/chat/completions --model judge-model
```

标准输入提示词列是 `Prompts`，可用 `--prompt-column` 改名。`api_batch_prompt_answer_generate.py` 保留 `Prompt` → `Answers` 的旧列契约；标准批量生成使用 `Responses`，裁判默认读取 `Responses` 并写 `Safety_Eval_Results`。使用不同列名时用 `--response-column` 显式指定。原始 ID 和标签列保留。

每条结果写入 `<output>.jsonl`，输出 CSV 通过原子替换生成，元数据写入 `<output>.meta.json`。相同来源下成功行跳过，失败行重试；来源/模型/参数改变需新输出路径。`status` 区分 `ok`、`error`、`skipped`，上游失败不会当成正常回答进入裁判。裁判仅接受完整 `Safe` / `UnSafe`。失败或跳过任务使命令非零退出。

`--timeout` 限制请求时间，`--max-attempts` 控制有界重试；计费服务可设置 `--max-attempts 1`。重试不提供服务端幂等保证。转换模块保留其 `OPENAI_BASE_URL` / `OPENAI_MODEL` / `OPENAI_API_KEY` 变量，详见模块说明；其 BASE_URL 为 SDK 服务根地址，与上述完整端点的语义不同。

## 图像、视频与训练

```powershell
python OpenClip-Chinese-Project/quick_start.py --model models/ChineseCLIP --image example.jpg --local-files-only
python base_ldm.py --smoke-test --device cpu
python full_ldm_unet.py --smoke-test --device cpu
python full_ldm_encode_decode.py --smoke-test --device cpu
python Text_and_Image_to_video_evaluate/batch_videos_frame_extract.py --input-dir videos --dry-run
```

LDM 的 `--smoke-test` 使用随机小张量，不下载 MNIST 或 VGG；完整训练才需要数据和预训练权重。CLIP 阈值属于实验参数，需要带标注验证集校准。抽帧和视频后端的参数、输入 JSON 与实际支持范围见[视频模块说明](Text_and_Image_to_video_evaluate/README.md)；云审核受理成功与最终审核结论需分别解释。

Notebook 需按单元顺序运行。图片生成在加载权重前规范化 ID 并检查重名、空提示词与种子范围；数据生成使用完整 JSON 解码，拒绝覆盖已损坏结果。Notebook 是探索入口，不提供生产评测完整的实验来源与自动续跑；正式文生图评测使用 `t2i_qwen_judge` 目录。

## 验证与迁移

```powershell
python tools/check_project.py
python tools/run_checks.py
```

检查工具只读取全部 Python 与 Notebook 代码，不导入或执行实验。凭据检查只报告位置，不回显值；它是轻量检查，不能保证识别所有凭据。`run_checks.py` 将公共测试与文生图评测回归放在独立进程运行，避免同名模块互相污染。可用 `--json-output docs/code_inventory.json` 保存静态文件清单。

原先自动执行、依赖绝对路径的脚本现在改为显式参数入口，旧的无参启动方式需按 `--help` 更新。旧输出缺来源元数据、输入或配置变更时，使用新输出目录。不要让多个任务同时写同一结果路径。已存在的 `t2i_qwen_judge` 管线和本地学习文件保持独立。

真实环境验收须记录：Python/依赖版本、模型 revision、输入数据摘要、GPU/CUDA、随机种子、完成/失败数量、总耗时及显存峰值。先用小样本检查图片/视频、原始回答和统计分母，再开展完整实验；并发、量化与编译加速在此基线上测量后逐项引入。
