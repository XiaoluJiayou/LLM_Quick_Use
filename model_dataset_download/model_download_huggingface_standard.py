from huggingface_hub import snapshot_download

# 1. 在这里填入你的 Hugging Face Access Token
# 获取地址：https://huggingface.co/settings/tokens
HF_TOKEN = ""

# 定义模型ID
model_id = 'stable-diffusion-v1-5/stable-diffusion-inpainting'
# 定义本地保存路径
save_path = "./autodl-tmp/models/stable-diffusion-v1-5/stable-diffusion-inpainting"
print(f"开始下载模型到: {save_path}")

model_dir = snapshot_download(
    repo_id=model_id,
    local_dir=save_path,
    # 国内镜像
    endpoint="https://hf-mirror.com",
    # 加入身份验证
    token=HF_TOKEN,
    # 可选：强制覆盖本地已存在的文件
    force_download=False,
    # 可选：遇到已存在文件跳过
    local_dir_use_symlinks=False
)

print(f"模型已保存至: {model_dir}")