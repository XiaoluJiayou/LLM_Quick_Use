#!/usr/bin/env python3
""" ChineseCLIP 组合词 NSFW 分类器（优化判定版）
核心调整：
1. 放宽双强匹配阈值，降低方差惩罚
2. 优化调整分计算（正常词基线分加权扣除，而非全额）
3. 调整组合分计算逻辑，提升得分友好性
4. 微调判定阈值，适配调整后的得分体系
"""
import argparse
import os, math
import statistics
from pathlib import Path
from typing import List
from llm_quick_use.clip import extract_features, load_clip

# ══════════════════════════════════════════════════════════════
# 配置区（优化后参数）
# ══════════════════════════════════════════════════════════════
MODEL_PATH = None
CLIP_TEMPLATE = "一张展示{}的照片"
IMAGE_PATHS = ["./imgs"]
SENSITIVE_WORDS = ["太庙婚礼", "故宫奔驰"]
NORMAL_WORDS = ["普通婚礼", "风景", "奔驰", "蓝天白云", "草地", "日常建筑", "普通汽车", "游客"]

# 判定阈值（调低，适配优化后的得分）
THRESHOLD = 0.01
# 子词最低过滤分（放宽）
MIN_TERM_SCORE = 0.55
# 乘积增益权重（提高，增强组合分）
PRODUCT_WEIGHT = 0.1
# 双强匹配高分阈值（放宽，从0.7→0.68）
HIGH_TERM_SCORE = 0.68
# 正常词基线分扣除权重（不再全额扣除，仅扣50%）
NORMAL_SCORE_WEIGHT = 0.5
# 方差惩罚系数（降低，从0.5→0.2）
VARIANCE_PENALTY = 0.2


# ══════════════════════════════════════════════════════════════
# 组合词拆分（保持不变）
# ══════════════════════════════════════════════════════════════
def split_word(word: str) -> List[str]:
    if not word:
        return []
    has_chinese = any('\u4e00' <= c <= '\u9fff' for c in word)
    if not has_chinese:
        return word.split() or [word]
    terms, buf = [], ""
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            buf += ch
        else:
            if buf:
                terms.extend([buf[i:i + 2] for i in range(0, len(buf), 2)])
                buf = ""
            if ch.strip():
                terms.extend(ch.split())
    if buf:
        parts = [buf[i:i + 2] if i + 2 <= len(buf) else buf[i:] for i in range(0, len(buf), 2)]
        if len(buf) % 2 == 1 and len(parts) >= 2:
            parts[-2] = parts[-2] + parts[-1]
            parts.pop()
        terms.extend(parts)
    return terms or [word]


# ══════════════════════════════════════════════════════════════
# 优化后的组合分计算逻辑
# ══════════════════════════════════════════════════════════════
def combined_score(scores: List[float]) -> float:
    if not scores:
        return 0.0

    # 规则1：所有子词≥最低分（已放宽）
    if min(scores) < MIN_TERM_SCORE:
        return 0.0

    # 规则2：双强匹配阈值（已放宽到0.68）
    if len(scores) == 2:
        t1, t2 = scores
        if not (t1 >= HIGH_TERM_SCORE and t2 >= HIGH_TERM_SCORE):
            return 0.0

    # 规则3：优化组合分计算
    # 1. 几何平均（核心得分）
    geo_avg = math.exp(sum(math.log(max(s, 1e-10)) for s in scores) / len(scores))
    # 2. 乘积增益（提高权重）
    product = 1.0
    for s in scores:
        product *= max(s, 1e-10)
    product_gain = product * PRODUCT_WEIGHT
    # 3. 方差惩罚（降低系数）
    variance = statistics.pvariance(scores) if len(scores) >= 2 else 0
    # 4. 最终组合分（减少惩罚，提高增益）
    final_score = geo_avg + product_gain - variance * VARIANCE_PENALTY
    return max(final_score, 0.0)


# ══════════════════════════════════════════════════════════════
# 分类器（优化调整分计算）
# ══════════════════════════════════════════════════════════════
class ClipClassifier:
    def __init__(self, model_path, template=CLIP_TEMPLATE, device=None, local_files_only=False):
        self.template = template
        self.model, self.processor = load_clip(model_path, device, local_files_only)

    def _encode_images(self, images):
        return extract_features(self.model, self.processor, images=images)

    def _encode_texts(self, texts):
        return extract_features(self.model, self.processor, texts=[self.template.format(t) for t in texts])

    def _similarity(self, img_feat, txt_feat):
        return ((img_feat @ txt_feat.T).cpu().numpy() + 1.0) / 2.0

    def classify(self, image_paths, sensitive_words, normal_words=None, batch_size=16):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not sensitive_words or any(not isinstance(w, str) or not w.strip() for w in sensitive_words):
            raise ValueError("Provide nonempty sensitive words")
        if not image_paths:
            raise ValueError("No supported images were found")
        results = []
        for start in range(0, len(image_paths), batch_size):
            results.extend(self._classify_batch(image_paths[start:start + batch_size], sensitive_words, normal_words))
        return results

    def _classify_batch(self, image_paths, sensitive_words, normal_words=None):
        from PIL import Image
        normal_words = normal_words or []
        images, valid_paths = [], []
        for p in image_paths:
            try:
                with Image.open(p) as source:
                    images.append(source.convert("RGB"))
                valid_paths.append(p)
            except Exception as e:
                raise ValueError("Cannot open image: {}".format(p)) from e
        if not images:
            return []

        word2terms = {w: split_word(w) for w in sensitive_words}
        all_sensitive_terms = list(dict.fromkeys([t for terms in word2terms.values() for t in terms]))
        filtered_normal_words = [w for w in normal_words if w not in all_sensitive_terms]
        all_texts = list(dict.fromkeys(all_sensitive_terms + filtered_normal_words))

        img_feat = self._encode_images(images)
        txt_feat = self._encode_texts(all_texts)
        sim = self._similarity(img_feat, txt_feat)
        txt_idx = {t: i for i, t in enumerate(all_texts)}

        results = []
        for i, path in enumerate(valid_paths):
            sens_scores = {}
            term_scores_all = {}

            for word, terms in word2terms.items():
                term_scores = [float(sim[i][txt_idx[t]]) for t in terms if t in txt_idx]
                term_scores_all[word] = term_scores
                sens_scores[word] = combined_score(term_scores)

            # 优化：正常词基线分仅扣除部分（不再全额减）
            max_normal = max((float(sim[i][txt_idx[w]]) for w in filtered_normal_words if w in txt_idx), default=0.0)
            # 调整分 = 敏感词组合分 - 正常词基线分 * 权重
            adj_scores = {w: s - (max_normal * NORMAL_SCORE_WEIGHT) for w, s in sens_scores.items()}

            best_word = max(adj_scores, key=adj_scores.get)
            best_adj = adj_scores[best_word]

            results.append({
                "image": str(path),
                "is_sensitive": best_adj > THRESHOLD,
                "best_word": best_word,
                "adjusted_score": round(best_adj, 4),
                "term_scores": term_scores_all[best_word],
                "raw_combined_score": round(sens_scores[best_word], 4),  # 新增：显示原始组合分
                "max_normal_score": round(max_normal, 4)  # 新增：显示正常词最高分
            })
        return results


def collect_images(path_list):
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    paths = []
    for p in path_list:
        p = Path(p)
        if p.is_file() and p.suffix.lower() in exts:
            paths.append(str(p))
        elif p.is_dir():
            paths.extend(str(f) for f in sorted(p.iterdir()) if f.is_file() and f.suffix.lower() in exts)
    return list(dict.fromkeys(paths))


def main(argv=None):
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--sensitive-words", nargs="+", default=SENSITIVE_WORDS)
    parser.add_argument("--normal-words", nargs="*", default=NORMAL_WORDS)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    image_paths = collect_images(args.images)
    if not image_paths:
        parser.error("No supported images were found")
    clf = ClipClassifier(args.model, device=args.device, local_files_only=args.local_files_only)
    results = clf.classify(image_paths, args.sensitive_words, args.normal_words, args.batch_size)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print("Heuristic scores are uncalibrated; validate thresholds on labeled data before safety use.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
