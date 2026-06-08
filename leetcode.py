import pandas as pd

# 以查看劫持基准测试为例
df_hijack = pd.read_json(r"D:\Desktop\llm-dataset\tensor-trust\benchmarks\extraction-robustness\v1\extraction_robustness_dataset.jsonl", lines=True)
print(f"提示词劫持extraction数据量: {len(df_hijack)} 条")

df_hijack = pd.read_json(r"D:\Desktop\llm-dataset\tensor-trust\benchmarks\hijacking-robustness\v1\hijacking_robustness_dataset.jsonl", lines=True)
print(f"提示词劫持hijacking数据量: {len(df_hijack)} 条")

df_detec = pd.read_json(r"D:\Desktop\llm-dataset\tensor-trust\detecting-extractions\v1\prompt_extraction_detection.jsonl", lines=True)
print(f"提示词劫持detection数据量: {len(df_detec)} 条")

# # 查看原始数据规模
# df_raw = pd.read_json(r"D:\Desktop\llm-dataset\tensor-trust\raw-data\v1\raw_dump_defenses.jsonl.bz2", lines=True,compression="bz2",).set_index("defense_id")
# print(f"原始对抗数据量: {len(df_raw)} 条")

# # 查看原始数据规模
# df_raw = pd.read_json(r"D:\Desktop\llm-dataset\tensor-trust\raw-data\v2\raw_dump_attacks.jsonl.bz2", lines=True,compression="bz2",).set_index("attack_id")
# print(f"原始对抗数据量: {len(df_raw)} 条")
#
# # 查看原始数据规模
# df_raw = pd.read_json(r"D:\Desktop\llm-dataset\tensor-trust\raw-data\v2\raw_dump_defenses.jsonl.bz2", lines=True,compression="bz2",).set_index("defense_id")
# print(f"原始对抗数据量: {len(df_raw)} 条")

