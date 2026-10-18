"""Generate a local video preview; despite the legacy filename this is a CLI."""
import argparse
from html import escape
from pathlib import Path
from image_base94_encode import encode_video_to_data_url, get_video_mime_type


def generate_html_for_video(video_path, output_html_path="play_video.html"):
    data_url = encode_video_to_data_url(video_path)
    title = escape(Path(video_path).name)
    mime = get_video_mime_type(video_path)
    html = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>视频预览: {title}</title><style>body{{background:#222;color:white;text-align:center}}video{{max-width:90%}}</style>
<h3>{title}</h3><video controls><source src="{data_url}" type="{mime}">浏览器不支持视频播放。</video></html>'''
    target = Path(output_html_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html, encoding="utf-8")
    return str(target.resolve())


def main(argv=None):
    parser = argparse.ArgumentParser(description="生成本地视频预览HTML（整个视频嵌入文件，适合小视频）")
    parser.add_argument("video")
    parser.add_argument("--output", default="play_video.html")
    args = parser.parse_args(argv)
    try:
        print(generate_html_for_video(args.video, args.output))
        return 0
    except (ValueError, OSError) as error:
        parser.exit(1, f"错误: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
