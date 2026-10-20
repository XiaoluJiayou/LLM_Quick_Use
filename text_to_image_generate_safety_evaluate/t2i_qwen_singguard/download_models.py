# Qwen/Qwen-Image-2.1 模型下载
from modelscope import snapshot_download

model_dir = snapshot_download(
    "Qwen/Qwen-Image-2.1",
    local_dir="/root/autodl-tmp/models/Qwen-Image-2.1",
)
print(model_dir)

# inclusionAI/Sing-Guard-8b 模型下载
model_dir = snapshot_download(
    "inclusionAI/Sing-Guard-8b",
    local_dir="/root/autodl-tmp/models/Sing-Guard-8b",
)
print(model_dir)
