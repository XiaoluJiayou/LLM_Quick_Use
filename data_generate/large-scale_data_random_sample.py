import pandas as pd


def stratified_sample_csv(input_file, output_file, total_sample_size):
    """
    从CSV文件中按一级领域的比例分层抽取数据。
    """
    print(f"正在读取文件: {input_file} ...")
    try:
        # 读取CSV文件，utf-8-sig可兼容中文乱码问题
        df = pd.read_csv(input_file, encoding='utf-8-sig')
    except FileNotFoundError:
        print(f"错误：找不到文件 {input_file}")
        return
    except Exception as e:
        print(f"读取文件出错: {e}")
        return

    # 检查必要的列是否存在
    required_columns = ['question', '一级领域', '二级领域']
    for col in required_columns:
        if col not in df.columns:
            print(f"错误：CSV文件中缺少必要的列 '{col}'")
            return

    # 检查数据量是否足够
    total_rows = len(df)
    if total_rows < total_sample_size:
        print(f"警告：原始数据总量({total_rows})少于目标抽样数({total_sample_size})，将返回所有数据。")
        sampled_df = df
    else:
        print(f"原始数据总量: {total_rows}")
        print(f"目标抽样数量: {total_sample_size}")

        # 1. 统计每个一级领域的原始数量
        category_counts = df['一级领域'].value_counts()

        # 2. 计算理论抽样数量（浮点数）
        theoretical_counts = (category_counts / total_rows) * total_sample_size

        # 3. 向下取整，得到基础分配数
        base_allocation = theoretical_counts.astype(int)

        # 4. 处理余数，补齐到 total_sample_size
        # 计算当前分配总数
        current_total = base_allocation.sum()
        deficit = total_sample_size - current_total

        # 如果有差额，根据小数部分大小进行补齐
        if deficit > 0:
            remainders = theoretical_counts - base_allocation
            # 找出小数部分最大的几个类别索引
            top_up_indices = remainders.nlargest(deficit).index

            # 对这些类别进行 +1
            # 注意：base_allocation 是 Series，可以直接通过索引修改
            for idx in top_up_indices:
                base_allocation[idx] += 1

        print("\n各领域抽样计划:")
        print(base_allocation)

        # 5. 分组抽样
        sampled_list = []

        for category, n_samples in base_allocation.items():
            # 筛选该类别数据
            group_data = df[df['一级领域'] == category]

            # 随机抽样 (random_state 保证结果可复现，可删除以获得不同结果)
            sampled_group = group_data.sample(n=n_samples, random_state=42)
            sampled_list.append(sampled_group)

        # 6. 合并结果
        sampled_df = pd.concat(sampled_list, ignore_index=True)

    # 7. 打乱最终数据顺序 (因为前面是按类别拼接的)
    sampled_df = sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 保存结果
    print(f"\n正在保存结果到: {output_file} ...")
    sampled_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print("处理完成！")
    print(f"最终数据行数: {len(sampled_df)}")
    print("最终一级领域分布验证:")
    print(sampled_df['一级领域'].value_counts())


# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    # 修改这里的文件名为你的实际文件名
    input_csv = r'F:\百万级多模态大模型安全评测数据集\北邮交付数据集_81.6W\文本样本_55W\dataset_55w.csv'
    output_csv = r'sampled_100.csv'
    target_count = 100

    stratified_sample_csv(input_csv, output_csv, target_count)
