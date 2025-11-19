#模型下载
from modelscope import snapshot_download
# model_dir = snapshot_download('Qwen/Qwen2.5-7B-Instruct', cache_dir="E:\\Qwen")
# model_dir = snapshot_download('xverse/XVERSE-7B-Chat', cache_dir="E:\\modelscope")

model_dir = snapshot_download('openai-community/gpt2', cache_dir="E:\\modelscope")