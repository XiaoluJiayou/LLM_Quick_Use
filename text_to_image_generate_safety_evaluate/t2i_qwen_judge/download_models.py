# Qwen/Qwen-Image-2.1 模型下载
from modelscope import snapshot_download

model_dir = snapshot_download(
    "Qwen/Qwen-Image-2.1",
    local_dir="/root/autodl-tmp/models/Qwen-Image-2.1",
)
print(model_dir)

# 裁判模型权重（上游仓库 inclusionAI/Sing-Guard-8b）
model_dir = snapshot_download(
    "inclusionAI/Sing-Guard-8b",
    local_dir="/root/autodl-tmp/models/t2i-judge",
)
print(model_dir)
