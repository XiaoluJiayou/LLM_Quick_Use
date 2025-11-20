import requests
import json
from tqdm import tqdm, trange
import urllib3
import pandas as pd
import os
import time


def main():
    # 数据集路径
    data_path_list = [r"C:\Users\unicom350\Desktop\ds_dataset\NIST Jailbreak Dataset\NIST Harmful Biology and Non-Biological Violent Activities Query Dataset.xlsx",
                      r"C:\Users\unicom350\Desktop\ds_dataset\NIST Jailbreak Dataset\NIST Jailbreak Dataset - 50 prompts.xlsx",
                      r"C:\Users\unicom350\Desktop\ds_dataset\NIST Jailbreak Dataset\NIST Malicious Hacking Query Dataset.xlsx",
                      r"C:\Users\unicom350\Desktop\ds_dataset\NIST Jailbreak Dataset\NIST Online Scamming Query Dataset - 50 prompts.xlsx"
                      ]
    # 测试的模型名
    model_name = "gpt-oss-20b"

    for data_path in data_path_list:
        # 根据文件地址，读取excel表格数据
        df = pd.read_excel(data_path, sheet_name='Sheet1', usecols=['Prompts'])
        df_len = len(df)
        # 仅提取文件名，不含扩展名
        data_name = os.path.basename(data_path)[:-5]
        # print(data_name)
        # print(df_len)
        # print(df.head())
        # print(df['Prompts'][15])
        # print(df.iloc[64]['Prompts'])
        # break

        # 调用url
        url = "https://qianfan.baidubce.com/v2/chat/completions"

        # 保存模型回复结果
        model_prompts_responses = []
        for i in tqdm(range(df_len)):
            # # 测试代码是否正常运行，并按照要求输出符合格式的结果
            # if i == 5:
            #     break
            prompt = df.iloc[i]['Prompts']
            payload = json.dumps({
                "model": "gpt-oss-20b",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant."
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "max_tokens": 256,
                "temperature": 0.9,
            })
            headers = {
                'Content-Type': 'application/json',
                'Authorization': ''
            }
            urllib3.disable_warnings()

            # 防止网络连接错误，或者api接口未响应等情况
            try:
                response = requests.request("POST", url, headers=headers, data=payload)
                # print(response.text)
                ans = json.loads(response.text)
                # print(ans['choices'][0]['message']['content'])
                ans_processing = ans['choices'][0]['message']['content']
                time.sleep(3)

            except:
                ans_processing = "裁判模型未响应"
                time.sleep(3)

            # 将模型指令和回复暂时存储
            model_prompts_responses.append([prompt, ans_processing])

        # 将结果保存为csv文件
        all_text_results = pd.DataFrame(model_prompts_responses, columns=['Prompts', 'Responses'])

        save_path = os.path.join("{}_data_{}.csv".format(model_name, data_name))
        print(save_path)
        all_text_results.to_csv(save_path, index=False)

        # # 测试使用
        # break




if __name__ == '__main__':
    main()