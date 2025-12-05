from huggingface_hub import hf_hub_download
import os

# 替换为您的 Hugging Face 用户访问令牌
HF_TOKEN = "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"

# 指定模型ID和文件名
repo_id = "CompVis/stable-diffusion-v-1-4-original"
filename = "sd-v1-4-full-ema.ckpt"

print(f"正在下载 {filename}...")

# 使用 hf_hub_download 函数下载文件
# use_auth_token 参数用于身份验证
downloaded_model_path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    token=HF_TOKEN,
    local_dir="D:\Desktop\llm_quick_use\model_dataset_download" # 可选：指定下载到的本地目录
)

print(f"模型已成功下载到: {os.path.abspath(downloaded_model_path)}")