import pandas as pd
import numpy as np
import random

# 正确设置随机种子以确保结果可重复
# 设置NumPy的随机种子
np.random.seed(42)
# 设置Python内置random模块的种子
random.seed(42)
# 在pandas的sample方法中也可以通过random_state参数设置

# 读取CSV文件（假设文件名为questions.csv）
data = pd.read_csv("F:\百万级多模态大模型安全评测数据集\北邮交付数据集_81.6W\文本样本_55W\dataset_55w.csv", usecols=['question'])

# 随机抽取1000行数据（无放回抽样）
sample_data = data.sample(n=1000, random_state=42)

# 将抽样结果保存到新文件
sample_data.to_csv('sampled_questions.csv', index=False)

print("抽样完成！已从55万行数据中随机抽取1000行数据。")
print(f"抽样数据已保存到sampled_questions.csv，包含{len(sample_data)}行数据。")