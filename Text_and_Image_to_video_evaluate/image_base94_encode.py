import os
import base64
from openai import OpenAI

# -----------------------------
# 1. 本地图像 → base64 → Data URL
# -----------------------------
def encode_image_to_data_url(image_path: str) -> str:
    """
    将本地图像转为 Data URL: data:image/{ext};base64,{base64_string}
    支持 jpeg/jpg/png/webp 等，按扩展名推断 MIME 类型。
    """
    # 读取二进制数据
    with open(image_path, "rb") as f:
        image_data = f.read()

    # 转为 base64 字符串
    base64_str = base64.b64encode(image_data).decode("utf-8")

    # 根据后缀推断 MIME 类型（可按需要补充）
    ext = os.path.splitext(image_path)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        mime_type = "image/jpeg"
    elif ext == ".png":
        mime_type = "image/png"
    elif ext == ".webp":
        mime_type = "image/webp"
    else:
        # 默认当 jpeg 处理，或根据你的实际图像格式修改
        mime_type = "image/jpeg"

    # 拼成 Data URL
    data_url = f"data:{mime_type};base64,{base64_str}"
    return data_url


# ------------------------------------------------
# 1. 核心功能：根据视频文件后缀获取 MIME 类型
# ------------------------------------------------
def get_video_mime_type(file_path: str) -> str:
    """
    根据文件扩展名返回标准的视频 MIME 类型。
    如果扩展名未知，则默认返回 'video/mp4'。
    """
    # 获取文件后缀并转为小写
    ext = os.path.splitext(file_path)[1].lower()

    # 常见视频格式的 MIME 类型映射表
    mime_map = {
        # MP4 及其变体
        ".mp4": "video/mp4",
        ".m4v": "video/mp4",
        ".m4p": "video/mp4",

        # MPEG 系列格式
        ".mpeg": "video/mpeg",
        ".mpg": "video/mpeg",
        ".mpe": "video/mpeg",

        # QuickTime / MOV
        ".mov": "video/quicktime",
        ".qt": "video/quicktime",

        # AVI
        ".avi": "video/x-msvideo",

        # WebM
        ".webm": "video/webm",

        # MKV
        ".mkv": "video/x-matroska",

        # FLV
        ".flv": "video/x-flv",

        # WMV
        ".wmv": "video/x-ms-wmv",

        # 3GPP
        ".3gp": "video/3gpp",
        ".3g2": "video/3gpp2",

        # 其他格式可根据需要在此补充
    }

    # 从映射表中查找，找不到则返回默认值
    return mime_map.get(ext, "video/mp4")


# ------------------------------------------------
# 2. 本地视频 -> Base64 -> Data URL
# ------------------------------------------------
def encode_video_to_data_url(video_path: str) -> str:
    """
    将本地视频文件转换为 Data URL 格式。
    """
    # 1. 读取视频二进制数据
    with open(video_path, "rb") as f:
        video_bytes = f.read()

    # 2. 转换为 Base64 字符串
    base64_str = base64.b64encode(video_bytes).decode("utf-8")

    # 3. 获取正确的 MIME 类型
    mime_type = get_video_mime_type(video_path)

    # 4. 拼接成 Data URL
    data_url = f"data:{mime_type};base64,{base64_str}"
    return data_url


def save_data_url_to_txt(data_url: str, output_txt_path: str):
    """
    将 Data URL 字符串保存到指定的 txt 文件中。

    :param data_url: 编码后的 Data URL 字符串 (例如 "data:image/png;base64,iVBORw0...")
    :param output_txt_path: 输出的 txt 文件路径 (例如 "output.txt")
    """
    try:
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(data_url)
        print(f"成功保存 Data URL 到: {os.path.abspath(output_txt_path)}")
    except Exception as e:
        print(f"保存失败: {e}")


def base64_to_image_video(data_url: str, output_path: str):
    """
    读取保存了 Data URL 的 txt 文件，将其解码还原为视频文件。
    """
    try:
        # 2. 提取 Base64 部分
        # 格式通常是 "data:video/mp4;base64,xxxxx"
        if "," in data_url:
            base64_str = data_url.split(",", 1)[1]
        else:
            # 如果只有纯 base64 字符串
            base64_str = data_url

        # 3. 解码并保存
        video_image_data = base64.b64decode(base64_str)

        with open(output_path, "wb") as f:
            f.write(video_image_data)

        print(f"解码成功！视频已保存至: {os.path.abspath(output_path)}")

    except Exception as e:
        print(f"解码失败: {e}")

# -----------------------------
# 3. 使用示例
# -----------------------------
if __name__ == "__main__":
    # local_image_path = r"C:\Users\unicom350\Downloads\24gf-ellipsis.png"         # 改成你本地的图像路径
    # print(local_image_path)
    # image_url = encode_image_to_data_url(local_image_path)
    # print(image_url)
    local_video_path = r"C:\Users\unicom350\Downloads\t2v-A14B_1280_720_1_Two_anthropomorphic_cats_in_comfy_boxing_gear_and__20260211_195642.mp4"
    # print(local_video_path)
    video_url = encode_video_to_data_url(local_video_path)
    # save_data_url_to_txt(video_url, "video_url.txt")
    base64_to_image_video(video_url, "copy.mp4")


