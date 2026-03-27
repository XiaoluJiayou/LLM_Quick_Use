import os


def generate_html_for_video(video_path: str, output_html_path: str = "play_video.html"):
    """
    读取本地视频，生成一个包含 Base64 视频的 HTML 文件，
    双击即可在浏览器中播放。
    """
    # 1. 读取视频并转 Base64 (复用之前的逻辑)
    import base64
    with open(video_path, "rb") as f:
        video_bytes = f.read()
    base64_str = base64.b64encode(video_bytes).decode("utf-8")

    # 简单推断 MIME 类型
    ext = os.path.splitext(video_path)[1].lower()
    mime_type = "video/mp4" if ext in [".mp4", ".m4v"] else "video/webm"  # 简化处理

    # 2. 构造 HTML 内容
    # 使用 <video> 标签直接读取 base64 数据
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>视频预览: {os.path.basename(video_path)}</title>
        <style>
            body {{ background-color: #222; color: white; text-align: center; padding-top: 50px; }}
            video {{ max-width: 90%; box-shadow: 0 0 20px rgba(0,0,0,0.5); }}
        </style>
    </head>
    <body>
        <h3>本地视频 Base64 预览</h3>
        <video controls autoplay>
            <source src="data:{mime_type};base64,{base64_str}" type="{mime_type}">
            您的浏览器不支持 video 标签。
        </video>
    </body>
    </html>
    """

    # 3. 保存 HTML 文件
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"成功生成 HTML 文件: {os.path.abspath(output_html_path)}")
    print("请双击该文件，使用浏览器打开以查看视频。")


# --- 使用示例 ---
if __name__ == "__main__":
    # 替换为你的视频路径
    local_video = r"C:\Users\unicom350\Downloads\t2v-A14B_1280_720_1_Two_anthropomorphic_cats_in_comfy_boxing_gear_and__20260211_195642.mp4"
    generate_html_for_video(local_video)
