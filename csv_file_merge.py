import pandas as pd
import glob


def merge_multiple_csv(input_dir, output_file):
    """
    合并目录下所有CSV文件（无需列名验证）

    参数:
        input_dir (str): 包含CSV文件的目录路径
        output_file (str): 合并后的输出文件路径
    """
    # 获取所有CSV文件路径
    all_files = glob.glob(f"{input_dir}/*.csv")
    if not all_files:
        raise FileNotFoundError(f"目录中未找到CSV文件: {input_dir}")

    # 读取所有文件并合并
    df_list = [pd.read_csv(file) for file in all_files]
    merged_df = pd.concat(df_list, ignore_index=True)

    # 保存结果
    merged_df.to_csv(output_file, index=False)
    print(f"成功合并 {len(all_files)} 个文件 → {output_file}")
    print(f"总行数: {len(merged_df)} | 列名: {list(merged_df.columns)}")


# 使用示例
if __name__ == "__main__":
    merge_multiple_csv(
        input_dir=r"C:\Users\unicom350\Desktop\quick_use\dataset",  # 替换为你的CSV目录
        output_file="merged_result.csv"  # 合并后的文件名
    )