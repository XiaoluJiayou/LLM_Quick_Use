import requests
import json
from pathlib import Path
from tqdm import tqdm, trange
import urllib3
import pandas as pd
import time

def main():
    url = "https://zwfwdmx.yardezx.com/open/api/model/v2/chat/completions"
    input_data_path = r"C:\Users\unicom350\Desktop\数据集\大语言模型数据集\model_safety_test_2k.xlsx"
    df = pd.read_excel(input_data_path, engine='openpyxl')
    text_results = []
    for i in tqdm(range(2500)):
        # if i == 10:
        #     break
        prompt = df.iloc[i]['Prompt']
        payload = json.dumps({
            "model": "qwen2.5-14b-gov",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 256,
            "temperature": 0.9,
        })
        headers = {
            'Content-Type': 'application/json',
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

        text_results.append([prompt, ans_processing])
        # time.sleep(0.5)

    # 将结果保存为csv文件
    all_text_results = pd.DataFrame(text_results, columns=['Prompts', 'Answers'])
    all_text_results.to_csv("gx_llm_safety_test_result.csv", index=False)


if __name__ == '__main__':
    main()
