import json
from pathlib import Path


def read_jsonl(file_path: str, encoding='utf-8') -> list[dict]:
    """
    安全读取包含中文的JSONL文件

    参数：
        file_path (str): 文件路径
        encoding (str): 文件编码，默认为utf-8

    返回：
        list[dict]: 包含所有解析结果的字典列表

    异常处理：
        - 文件不存在时抛出FileNotFoundError[2](@ref)
        - JSON解析错误时跳过无效行并打印警告[3](@ref)
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"文件 {file_path} 不存在")

    data = []
    with open(file_path, 'r', encoding=encoding) as f:
        for line_num, line in enumerate(f, 1):
            stripped_line = line.strip()
            if not stripped_line:
                continue  # 跳过空行[7](@ref)

            try:
                # 解析中文内容时禁用ASCII转义[4,7](@ref)
                item = json.loads(stripped_line, strict=False)
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"警告：第{line_num}行解析失败 - {e}")

    return data


# 使用示例
if __name__ == "__main__":
    try:
        # 读取包含中文的JSONL文件
        results = read_jsonl("E:\\Downloads\\dataset\\firefly_pretrain-dataset\\CNewSum_v2.jsonl")

        # 打印前2条数据验证
        for idx, item in enumerate(results[:5], 1):
            print(f"记录{idx}:")
            print("-" * 40)
            print(item["text"][:-1] + "...")
            print(f"总字数：{len(item['text'])}")
            print("-" * 40 + "\n")

    except Exception as e:
        print(f"运行错误：{str(e)}")