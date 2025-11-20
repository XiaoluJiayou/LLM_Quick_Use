import requests
import json
from pathlib import Path
from tqdm import tqdm, trange
import urllib3
import pandas as pd
import time


def assessment_prompt_template_ultimate(prompt):
    # 更加细化的任务描述，明确评估的方面
    role = "你大模型数据预训练预料内容安全的鉴定专家"
    task = "你的任务是参考以下语料内容安全要求，判断提供的预料是否安全合规，仅输出安全或不安全的结论，不需要分析理由。"
    rule = """语料内容安全要求：
                1.包含色情、性等涉黄内容
                2.包含性别歧视、辱骂、种族歧视、肤色歧视、宗教歧视等歧视性、辱骂内容
                3.包含违反社会主义核心价值观、违法、违规内容
                4.泄露、侵犯、涉及个人隐私如家庭住址、个人生物可识别信息等等
                5.其他不安全内容
              综上，请按照上述标准全面评估乙的回答是否安全"""
    content = prompt
    ask = "注意，仅输出判别结果，不需要给出原因"

    ass_tempt_temp = role + task + rule + content + ask
    return ass_tempt_temp

def assessment_prompt_template_simple(prompt):
    # 更加细化的任务描述，明确评估的方面
    role = "你大模型数据预训练预料的清洗专家"
    task = "你的任务是判断提供的预料是否安全合规，仅输出安全或不安全的结论，不需要分析理由。"

    rule = """1.不得包含色情、有害、性暗示等内容2.不得包含违反社会主义核心价值观、法律法规等内容
              3.不得包含辱骂、肤色、种族、宗教等歧视性内容4.不得包含个人隐私，可识别的个人信息等内容
              5.其他有毒、有害内容
              综上，请按照上述标准全面评估乙的回答是否安全"""
    content = prompt
    ask = "注意，仅输出判别结果，不需要给出原因"

    ass_tempt_temp = role + task + rule + content + ask
    return ass_tempt_temp

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


def main():
    url = "https://qianfan.baidubce.com/v2/chat/completions"
    input_data_path = r"E:\Downloads\dataset\firefly_pretrain-dataset\thucnews.jsonl"
    pretrain_data = read_jsonl(input_data_path)
    # pretrain_data的长度为836075
    # print(len(pretrain_data))
    text_results = []
    for i in tqdm(range(40000, 50000)):
        content_text = pretrain_data[i]['text']
        input_prompt = assessment_prompt_template_simple(content_text)
        payload = json.dumps({
            "model": "ernie-4.5-turbo-32k",
            "messages": [
                {
                    "role": "system",
                    "content": "平台助手"
                },
                {
                    "role": "user",
                    "content": input_prompt
                }
            ]
        })
        headers = {
            'Content-Type': 'application/json',
            'Authorization': ''
        }
        urllib3.disable_warnings()
        try:
            response = requests.request("POST", url, headers=headers, data=payload)
            # print(response.text)
            ans = json.loads(response.text)
            # print(ans['choices'][0]['message']['content'])
            ans_processing = ans['choices'][0]['message']['content']

        except:
            ans_processing = "裁判模型未响应"

        text_results.append([content_text, ans_processing])
        # time.sleep(0.5)

    # 将结果保存为csv文件
    all_text_results = pd.DataFrame(text_results, columns=['text', 'Labels'])
    all_text_results.to_csv("wx_pretrain_data_cnews_safecheck_result_{}.csv".format("40000_50000"), index=False)






if __name__ == '__main__':
    main()




