import requests


def analyze_response(response):
    """完整分析响应对象"""

    # 1. 基本状态信息
    print("=== 基本状态 ===")
    print(f"状态码: {response.status_code}")
    print(f"状态描述: {response.reason}")
    print(f"请求是否成功: {response.ok}")
    print(f"耗时: {response.elapsed}")

    # 2. URL和重定向信息
    print("\n=== URL信息 ===")
    print(f"最终URL: {response.url}")
    print(f"重定向历史: {len(response.history)} 次")
    for i, resp in enumerate(response.history):
        print(f"  重定向 {i + 1}: {resp.status_code} -> {resp.url}")

    # 3. 头部信息
    print("\n=== 响应头 ===")
    for key, value in response.headers.items():
        print(f"{key}: {value}")

    # 4. 内容信息
    print("\n=== 内容信息 ===")
    print(f"编码: {response.encoding}")
    print(f"内容长度: {len(response.content)} 字节")
    print(f"文本预览: {response.text[:200]}...")

    # 5. 尝试解析JSON
    try:
        json_data = response.json()
        print("\n=== JSON数据 ===")
        print(f"JSON类型: {type(json_data)}")

        # 如果有conclusion字段
        if "conclusion" in json_data:
            print(f"结论: {json_data['conclusion']}")

        # 打印所有键
        print(f"JSON键: {list(json_data.keys())}")
    except ValueError:
        print("响应不是有效的JSON格式")

    # 6. 请求信息
    print("\n=== 原始请求信息 ===")
    print(f"请求方法: {response.request.method}")
    print(f"请求头: {dict(response.request.headers)}")
    if response.request.body:
        print(f"请求体长度: {len(response.request.body)} 字节")


# 使用示例
def main():
    url = "https://aip.baidubce.com/rest/2.0/solution/v1/video_censor/v2/user_defined"
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    payload = 'extId=123&name=test&videoUrl=http://example.com/video.mp4'

    try:
        response = requests.request("POST", url, headers=headers, data=payload.encode("utf-8"))
        response.encoding = "utf-8"

        # 完整分析响应
        analyze_response(response)

    except requests.exceptions.RequestException as e:
        print(f"请求异常: {e}")


if __name__ == '__main__':
    main()