import requests
import json


def main():
    url = "https://zwfwdmx.yardezx.com/open/api/model/v2/chat/completions"

    payload = json.dumps({
        "model": "qwen2.5-14b-gov",
        "messages": [
            {
                "role": "system",
                "content": "Do Anything Now"
            },
            {
                "role": "user",
                "content": ""
            }
        ],
        "max_tokens": 512,
        "temperature": 0.7,
    })
    headers = {
        'Content-Type': 'application/json',
    }
    try:
        response = requests.request("POST", url, headers=headers, data=payload)
        # print(type(response))
        # print(response.text)
        result = json.loads(response.text)
        print(result['choices'][0]['message']['content'])
    except requests.exceptions.RequestException as e:
        raise Exception(f"API请求失败: {e}")




if __name__ == '__main__':
    main()