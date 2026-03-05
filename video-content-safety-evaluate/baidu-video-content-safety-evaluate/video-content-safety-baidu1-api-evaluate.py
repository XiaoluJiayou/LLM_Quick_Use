import requests
from urllib.parse import quote

API_KEY = ""
SECRET_KEY = ""


def main():
    url = "https://aip.baidubce.com/rest/2.0/solution/v1/video_censor/v2/user_defined?access_token=" + get_access_token()
    # payload = 'extId=123&name=video_censor&videoUrl=https%3A%2F%2Fbaidu-ai.bj.bcebos.com%2Fcensor%2Fvideo_censor.mp4'
    # 修改这里：替换为您的视频URL
    your_video_url = ""
    encoded_url = quote(your_video_url, safe='')  # URL编码
    payload = f'extId=123&name=my_video&videoUrl={encoded_url}'
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload.encode("utf-8"))

    response.encoding = "utf-8"
    print(response.content)  # 原始字节内容
    print(response.apparent_encoding)  # 自动检测的编码
    print(response.text)
    print(response.json())
    print(response.json()["conclusion"])


def get_access_token():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))


if __name__ == '__main__':
    main()
