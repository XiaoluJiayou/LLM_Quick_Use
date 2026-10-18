"""Strict media Data URL round-trips. No API client or network dependency."""
import argparse
import base64
import binascii
import mimetypes
from pathlib import Path

VIDEO_MIMES = {".mp4": "video/mp4", ".m4v": "video/mp4", ".mpeg": "video/mpeg",
               ".mpg": "video/mpeg", ".mov": "video/quicktime", ".avi": "video/x-msvideo",
               ".webm": "video/webm", ".mkv": "video/x-matroska", ".flv": "video/x-flv",
               ".wmv": "video/x-ms-wmv", ".3gp": "video/3gpp", ".3g2": "video/3gpp2"}
IMAGE_MIMES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
               ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp"}


def get_video_mime_type(file_path):
    try:
        return VIDEO_MIMES[Path(file_path).suffix.lower()]
    except KeyError:
        raise ValueError("不支持的视频格式") from None


def _encode(path, mime):
    payload = Path(path).read_bytes()
    if not payload:
        raise ValueError("媒体文件为空")
    return "data:" + mime + ";base64," + base64.b64encode(payload).decode("ascii")


def encode_image_to_data_url(image_path):
    try:
        mime = IMAGE_MIMES[Path(image_path).suffix.lower()]
    except KeyError:
        raise ValueError("不支持的图片格式") from None
    return _encode(image_path, mime)


def encode_video_to_data_url(video_path):
    return _encode(video_path, get_video_mime_type(video_path))


def save_data_url_to_txt(data_url, output_txt_path):
    Path(output_txt_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_txt_path).write_text(data_url, encoding="utf-8")


def base64_to_image_video(data_url, output_path):
    value = data_url.strip()
    if value.startswith("data:"):
        header, sep, value = value.partition(",")
        if not sep or not header.endswith(";base64"):
            raise ValueError("Data URL 必须采用base64编码")
    try:
        data = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("无效的base64内容") from None
    if not data:
        raise ValueError("解码内容为空")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_bytes(data)
    return str(output_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description="媒体文件与Data URL转换")
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--decode", action="store_true", help="输入为Data URL文本文件")
    args = parser.parse_args(argv)
    try:
        if args.decode:
            base64_to_image_video(Path(args.input).read_text(encoding="utf-8"), args.output)
        else:
            encode = encode_video_to_data_url if Path(args.input).suffix.lower() in VIDEO_MIMES else encode_image_to_data_url
            save_data_url_to_txt(encode(args.input), args.output)
    except (OSError, ValueError) as error:
        parser.exit(1, f"错误: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
