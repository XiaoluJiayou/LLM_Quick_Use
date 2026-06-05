# from huggingface_hub import snapshot_download
#
# snapshot_download(repo_id="knoveleng/redbench", repo_type="dataset", local_dir="D:\\Desktop\\llm-dataset\\redbench", resume_download=True)

import os

# 设置镜像源（在 import huggingface_hub 之前设置）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="qxcv/tensor-trust",
    repo_type="dataset",
    local_dir="D:\\Desktop\\llm-dataset\\tensor-trust",
    resume_download=True
)




# import pandas as pd
#
# # 读取 parquet 文件
# df = pd.read_parquet(r"D:\Desktop\llm-dataset\JailbreakHub_1.5W\data\train-00000-of-00001.parquet", engine='fastparquet')
#
# # 查看前几行数据
# print(df.head())
# print(df[:1])
