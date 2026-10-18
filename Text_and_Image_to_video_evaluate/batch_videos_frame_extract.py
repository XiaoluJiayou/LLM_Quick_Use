"""Extract bounded, evenly spaced frames spanning each whole video."""
import argparse
import json
import math
import os
from pathlib import Path
import tempfile
from _video_workflow import IMAGE_SUFFIXES, VIDEO_SUFFIXES, atomic_json, file_digest, find_files, positive_number


def sampling_indices(total_frames, video_fps, extract_fps=1, max_frames=30):
    positive_number(video_fps, "video_fps")
    if not isinstance(total_frames, int) or total_frames < 1:
        raise ValueError("视频总帧数必须为正整数")
    if not isinstance(max_frames, int) or isinstance(max_frames, bool) or max_frames < 1:
        raise ValueError("max_frames 必须为正整数")
    if extract_fps is None:
        count = min(total_frames, max_frames)
    else:
        positive_number(extract_fps, "extract_fps")
        count = min(total_frames, max_frames, max(1, math.ceil(total_frames / video_fps * extract_fps)))
    if count == 1:
        return [total_frames // 2]
    return [round(i * (total_frames - 1) / (count - 1)) for i in range(count)]


def extract_single_video(video_path, save_dir, extract_fps=1, max_frames=30, overwrite=False):
    """Stage decoded frames before replacing outputs; require explicit overwrite."""
    if extract_fps is not None:
        positive_number(extract_fps, "extract_fps")
    if not isinstance(max_frames, int) or isinstance(max_frames, bool) or max_frames < 1:
        raise ValueError("max_frames 必须为正整数")
    source, destination = Path(video_path), Path(save_dir)
    if not source.is_file():
        raise FileNotFoundError(f"视频不存在: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    old_frames = find_files(destination, IMAGE_SUFFIXES)
    if old_frames and not overwrite:
        raise FileExistsError(f"帧目录非空: {destination}；请使用新目录或 --overwrite")
    import cv2
    cap = cv2.VideoCapture(str(source))
    try:
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {source}")
        raw_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        positive_number(raw_count, "视频总帧数")
        total_frames = int(raw_count)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        indices = sampling_indices(total_frames, video_fps, extract_fps, max_frames)
        source_digest = file_digest(source)
        with tempfile.TemporaryDirectory(prefix=".extract-", dir=destination) as staging:
            targets, names, frame_count = set(indices), [], 0
            while frame_count <= indices[-1]:
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f"视频提前解码失败: 第{frame_count}帧，共声明{total_frames}帧")
                if frame_count in targets:
                    name = f"{len(names) + 1:03d}.jpg"
                    if not cv2.imwrite(str(Path(staging) / name), frame):
                        raise OSError(f"无法保存抽帧图片: {name}")
                    names.append(name)
                frame_count += 1
            if len(names) != len(indices):
                raise ValueError("抽帧未完成")
            if file_digest(source) != source_digest:
                raise ValueError("抽帧期间源视频发生变化，请重试")
            # Deletion is confined to image files explicitly discovered in this output directory.
            atomic_json(destination / "frames_manifest.json", {"status": "updating"})
            if overwrite:
                for old in old_frames:
                    old.unlink()
            for name in names:
                os.replace(Path(staging) / name, destination / name)
            manifest = {"version": 1, "status": "complete",
                        "frame_sha256": {name: file_digest(destination / name) for name in names}, "source": str(source.resolve()), "sha256": source_digest,
                        "total_frames": total_frames, "video_fps": video_fps,
                        "extract_fps": extract_fps, "max_frames": max_frames,
                        "sampled_indices": indices, "frame_files": names,
                        "saved_count": len(names), "duration": total_frames / video_fps,
                        "sampling": "uniform_whole_video",
                        "interval": (indices[-1] - indices[0]) / (len(indices) - 1) if len(indices) > 1 else 0}
            atomic_json(destination / "frames_manifest.json", manifest)
            return manifest
    finally:
        cap.release()


def batch_extract_frames(input_dir=None, output_dir=None, extract_fps=1, max_frames=30, overwrite=False):
    base = Path(__file__).parent
    input_dir = input_dir or base / "Videos"
    output_dir = Path(output_dir or base / "Videos-Frames")
    if extract_fps is not None:
        positive_number(extract_fps, "extract_fps")
    videos = find_files(input_dir, VIDEO_SUFFIXES)
    if not videos:
        raise ValueError("输入目录没有视频")
    names = [p.stem.casefold() for p in videos]
    if len(set(names)) != len(names):
        raise ValueError("视频文件具有相同名称，输出目录会冲突；请先重命名")
    results = []
    for video in videos:
        try:
            result = extract_single_video(video, output_dir / (video.stem + "_Frames"),
                                          extract_fps, max_frames, overwrite)
            results.append({"video": str(video), "status": "success", **result})
        except (OSError, ValueError) as error:
            results.append({"video": str(video), "status": "failed", "error": str(error)})
    atomic_json(output_dir / "extraction_results.json", results)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description="覆盖完整视频的均匀抽帧；旧帧需要显式 --overwrite")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--extract-fps", type=float, default=1)
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        positive_number(args.extract_fps, "extract_fps")
        if args.max_frames < 1:
            raise ValueError("max_frames 必须为正整数")
        if args.dry_run:
            videos = find_files(args.input_dir or Path(__file__).parent / "Videos", VIDEO_SUFFIXES)
            if not videos:
                raise ValueError("输入目录没有视频")
            print(json.dumps([str(p) for p in videos], ensure_ascii=False))
            return 0
        results = batch_extract_frames(args.input_dir, args.output_dir, args.extract_fps,
                                       args.max_frames, args.overwrite)
        print(f"完成: {len(results)}，失败: {sum(r['status'] != 'success' for r in results)}")
        return int(any(r["status"] != "success" for r in results))
    except (OSError, ValueError, ImportError) as error:
        parser.exit(1, f"错误: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
