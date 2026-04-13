import json

# 读取原始JSON数据（假设数据保存在T2VSafetyBench_sampled.json文件中）
with open(r"D:\Desktop\数据集\文生视频数据集\T2VSafetyBench_sampled.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 重新分配index，从1开始递增
for new_index, item in enumerate(data, start=1):
    # 确保新index不超过120（如果原始数据超过120条，只处理前120条）
    if new_index > 120:
        break
    item["index"] = new_index

# 处理原始数据不足120条的情况（可选：补充空条目至120条）
while len(data) < 120:
    data.append({
        "index": len(data) + 1,
        "prompt": "",
        "category": ""
    })

# 将修改后的数据写入新文件（或覆盖原文件）
with open(r"D:\Desktop\数据集\文生视频数据集\T2VSafetyBench_sampled_new.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("索引已重新编排为1-120，结果已保存至T2VSafetyBench_sampled_updated.json")