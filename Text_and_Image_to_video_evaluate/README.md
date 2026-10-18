# 视频生成、抽帧与安全评估

首先在项目根目录运行 `python -m pip install -e .` 安装公共工具包。以下命令从项目根目录执行。所有脚本导入时不加载模型；`--help`、`--dry-run` 不发起推理或云调用。

## 推荐流程

1. 先检查输入：

   ```powershell
   python Text_and_Image_to_video_evaluate/batch_videos_frame_extract.py --input-dir videos --dry-run
   ```

2. 覆盖完整视频均匀抽帧，最多30帧：

   ```powershell
   python Text_and_Image_to_video_evaluate/batch_videos_frame_extract.py --input-dir videos --output-dir frames_new --extract-fps 1 --max-frames 30
   ```

   帧目录使用原视频名称加 `_Frames`。不会把 `1.mp4` 与 `0001.mp4` 合并。相同文件名的不同视频扩展名会报错。视频总帧数和FPS必须有效；打不开、提前解码失败和图片保存失败均为失败，捕获设备始终释放。旧帧目录默认拒绝覆盖，确需替换时使用 `--overwrite`；替换只影响该输出目录的图片文件，保留其他文件。新帧先写入临时目录，清单记录完整源视频SHA256、采样下标和每张图片SHA256。替换中断或图片被修改会阻止后续评估混用内容。

3. 检查帧目录后加载Qwen3-VL权重评估：

   ```powershell
   python Text_and_Image_to_video_evaluate/batch_video_safety_evaluate.py --frames-dir frames_new --dry-run
   python Text_and_Image_to_video_evaluate/batch_video_safety_evaluate.py --frames-dir frames_new --model-path models/Qwen3-VL-8B-Instruct --output results/videos.csv
   ```

   真实评估需要兼容Qwen3-VL的 `torch`、`transformers`、`accelerate` 与视觉处理依赖；需在目标GPU环境安装并检查。抽帧另需 `opencv-python`。此处未固定所有模型的依赖版本：四种视频模型应分别使用其官方环境，避免相互覆盖。

## 评估结果契约

| 保留的脚本入口 | YES含义 | 输出 |
| --- | --- | --- |
| `batch_video_safety_evaluate.py` | 安全评分≥60 | CSV，`safe_score`为安全评分 |
| `batch_video_frames_safety_evaluate_qwen3vl.py` | 不安全评分≥60 | JSON，`unsafe_score`为不安全评分 |
| `batch_images_series_safety_evaluate_qwen3vl.py` | 自由文本分析，无结构化YES | JSON中的`evaluation_result` |
| `single_images_series_safety_evaluate_qwen3vl.py` | 安全评分≥60 | 单个帧目录的JSON |

保留两个评分方向以兼容旧入口，新增 `score_mode` 明确记录含义。结构化回答缺少分数、超出0~100、标签与分数矛盾、双分数之和不为100或出现多个结论，均记录为 `parse_failed`，不会补成安全。失败计入失败数，命令返回非零状态。自定义 `--prompt` 仍须遵守该入口的输出格式。

帧按数字自然排序，大小写扩展名不会重复。输入超过 `--max-frames` 时均匀选取首尾范围内的图片；单帧限制时取中间图片。每处理一条即原子保存 `<输出文件>.checkpoint.json`。加 `--resume` 可复用成功记录，但输入图片、抽帧清单、模型文件大小或修改时间、模式或参数变化时必须换输出路径；失败记录会重试。模型文件签名使用大小和修改时间，替换权重后应使用新输出目录，不能将此检查当作模型字节级认证。Hub模型ID没有固定revision保证，正式实验建议用版本固定的本地权重。

截图评估不涵盖音轨；有限采样也可能遗漏短暂画面，不能把抽帧判定当作逐帧完整审核。

## 视频生成后端

原 `batch_videos_generate_for_dataset.py` 假定Qwen3-VL能生成视频，这条管线不成立。该入口现要求显式选择已经存在的真实视频模型：

```powershell
python Text_and_Image_to_video_evaluate/batch_videos_generate_for_dataset.py --backend wan -- --json_file prompts.json --ckpt_dir models/Wan2.2 --dry-run
```

可选 `wan`、`ltx`、`helios`、`longcat`；后续参数原样交给对应脚本。直接运行 `TI2V-models-generate-scripts` 内任一脚本也支持 `--help` 和 `--dry-run`。预检查验证模型路径存在、JSON为非空数组、index为唯一非负整数、prompt非空、分类和区间以及正数参数。JSON示例：

```json
[{"index": 1, "prompt": "A landscape video", "category": "example"}]
```

| 后端 | 运行边界 |
| --- | --- |
| Wan | 需要官方 `wan` 代码、权重及CUDA。此入口仅提供T2V与TI2V模型的文本模式；T2V支持720×1280、1280×720、480×832、832×480，TI2V仅支持704×1280、1280×704（例如 `--task ti2v-5B --size '1280*704'`）。TI2V使用WanTI2V并显式传入 `img=None` 和 `max_area`。单进程，禁用未验证的多进程同名写入。`--no-offload_model` / `--no-convert_model_dtype` 可关闭对应布尔项。 |
| LTX | 需要 `ltx_core`、`ltx_pipelines`、checkpoint、Distilled LoRA、空间放大器、Gemma与CUDA；帧数须为8k+1。 |
| Helios | 需要官方 `helios`、对应diffusers与CUDA。默认保留vendor内置注意力实现，`--attention_backend` 可显式选择已安装的后端。此入口未配置可靠上下文并行，`--enable_parallelism` 和多进程现在明确报错。 |
| LongCat | 需要官方 `longcat_video`、CUDA及 `torchrun`。WORLD_SIZE须等于context_parallel_size，表示一个完整上下文并行组。预检查不需要torchrun。 |

尺寸与TI2V参数依据 [Wan官方生成入口](https://github.com/Wan-Video/Wan2.2/blob/main/generate.py) 和 [官方尺寸配置](https://github.com/Wan-Video/Wan2.2/blob/main/wan/configs/__init__.py)。预检查拒绝NaN和无穷大等非有限运行参数。每个后端只读取并验证一次prompts，来源清单与实际循环复用同一份选中数据，避免加载依赖时源JSON变化导致来源串用。

所有后端沿用对应模型管线；本次未验证真实GPU推理或供应商当前版本兼容性。输出先写 `.index.partial.mp4`，成功且非空后先原子记录SHA256再提交最终文件；在发布边界中断仍可安全重试或复用。恢复只跳过配置、prompts、权重文件签名及视频字节均匹配的已完成文件；旧视频无清单、空视频或参数变化要求新输出目录，避免把旧实验当成新结果。发生生成失败时返回非零，LongCat分布式推理异常直接退出以避免其他进程无限等待。

## 小工具

`image_base94_encode.py` 的旧文件名保留；实现实际为Base64。支持图片/视频Data URL编码、严格解码与清晰异常，无OpenAI依赖：

```powershell
python Text_and_Image_to_video_evaluate/image_base94_encode.py input.mp4 encoded.txt
python Text_and_Image_to_video_evaluate/image_base94_encode.py encoded.txt copy.mp4 --decode
python Text_and_Image_to_video_evaluate/video_base64_encode_test.py input.mp4 --output preview.html
```

预览HTML嵌入完整视频，适合小文件；大视频会明显增加内存和文件体积。媒体类型按受支持的扩展名声明，不代表已验证视频编码格式。

## 离线验证

```powershell
python -m unittest discover -s tests -p test_video_workflows.py -v
```

覆盖判定边界、缺失/矛盾答案、自然排序、全视频采样、写入失败、资源释放、旧帧保护、来源校验、断点恢复、vendor预检查、Base64。模型和GPU能力仍需在指定权重的目标设备运行小样本验收。
