#!/usr/bin/env python3
""" ChineseCLIP 组合词 NSFW 分类器（优化判定版）
核心调整：
1. 放宽双强匹配阈值，降低方差惩罚
2. 优化调整分计算（正常词基线分加权扣除，而非全额）
3. 调整组合分计算逻辑，提升得分友好性
4. 微调判定阈值，适配调整后的得分体系
"""
import os, math
from pathlib import Path
from typing import List
import numpy as np, torch
from PIL import Image
from transformers import ChineseCLIPModel, ChineseCLIPProcessor

# ══════════════════════════════════════════════════════════════
# 配置区（优化后参数）
# ══════════════════════════════════════════════════════════════
MODEL_PATH = "/root/autodl-tmp/chinese-clip-vit-base-patch16"
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
    variance = np.var(scores) if len(scores) >= 2 else 0
    # 4. 最终组合分（减少惩罚，提高增益）
    final_score = geo_avg + product_gain - variance * VARIANCE_PENALTY
    return max(final_score, 0.0)


# ══════════════════════════════════════════════════════════════
# 分类器（优化调整分计算）
# ══════════════════════════════════════════════════════════════
class ClipClassifier:
    def __init__(self, model_path=MODEL_PATH, template=CLIP_TEMPLATE):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.template = template
        print(f"[加载模型] {model_path} → {self.device}")
        self.model = ChineseCLIPModel.from_pretrained(model_path).to(self.device).eval()
        self.processor = ChineseCLIPProcessor.from_pretrained(model_path)

    @torch.no_grad()
    def _encode_images(self, images):
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
        feat = self.model.get_image_features(**inputs)
        if hasattr(feat, "pooler_output"):
            feat = feat.pooler_output
        return feat / feat.norm(p=2, dim=-1, keepdim=True)

    @torch.no_grad()
    def _encode_texts(self, texts):
        templated = [self.template.format(t) for t in texts]
        inputs = self.processor(text=templated, padding=True, truncation=True, return_tensors="pt").to(self.device)
        feat = self.model.get_text_features(**inputs)
        if hasattr(feat, "pooler_output"):
            feat = feat.pooler_output
        return feat / feat.norm(p=2, dim=-1, keepdim=True)

    def _similarity(self, img_feat, txt_feat):
        return ((img_feat @ txt_feat.T).cpu().numpy() + 1.0) / 2.0

    def classify(self, image_paths, sensitive_words, normal_words=None):
        normal_words = normal_words or []
        images, valid_paths = [], []
        for p in image_paths:
            try:
                images.append(Image.open(p).convert("RGB"))
                valid_paths.append(p)
            except Exception as e:
                print(f" ⚠ 跳过: {p} ({e})")
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
                "image": os.path.basename(path),
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
    return paths


def main():
    print(f"\n{'=' * 60}")
    print(f" ChineseCLIP 组合词 NSFW 分类器（优化判定版）")
    print(f"{'=' * 60}")
    print(f" 敏感词: {SENSITIVE_WORDS}")
    print(f" 双强匹配阈值: {HIGH_TERM_SCORE} (两个子词必须同时≥此分才敏感)")
    print(f" 判定阈值: {THRESHOLD} | 正常词扣除权重: {NORMAL_SCORE_WEIGHT}")
    print(f"{'=' * 60}")

    image_paths = collect_images(IMAGE_PATHS)
    clf = ClipClassifier()
    results = clf.classify(image_paths, SENSITIVE_WORDS, NORMAL_WORDS)

    print(f"\n{'=' * 60}")
    for r in results:
        tag = "🔴 敏感" if r["is_sensitive"] else "🟢 正常"
        print(f"{r['image']}: {tag}")
        print(f"   匹配词: {r['best_word']} | 调整分: {r['adjusted_score']:.4f}")
        print(f"   原始组合分: {r['raw_combined_score']:.4f} | 正常词最高分: {r['max_normal_score']:.4f}")
        print(f"   子词分数: {[round(s, 4) for s in r['term_scores']]}")
        print(f"   {'-' * 50}")
    print(f"{'=' * 60}")
    cnt = sum(r["is_sensitive"] for r in results)
    print(f"汇总: {len(results)} 张中 {cnt} 张敏感")


if __name__ == "__main__":
    main()