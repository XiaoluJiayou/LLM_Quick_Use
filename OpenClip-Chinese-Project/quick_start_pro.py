from PIL import Image
import torch
from transformers import ChineseCLIPProcessor, ChineseCLIPModel


# ========================== 配置区（只需修改这里） ==========================

# 模型路径
MODEL_PATH = "/root/autodl-tmp/chinese-clip-vit-base-patch16"

# CLIP标准提示模板（中文最优格式，可根据需求微调）
CLIP_TEMPLATE = "一张展示{}的照片"

# 待分类图像路径
IMAGE_PATH = "./image6.jpg"

# 4个精细对照组：同物体、不同场景，区分安全/不安全
CLASS_PROMPTS = {
    # 不安全：故宫里的奔驰
    "unsafe_mercedes_in_forbidden_city": [
        "两个女子开这奔驰汽车停放在故宫的广场上",
        "太和殿广场有两个女生开的奔驰私家车",
    ],
    # 不安全：太庙举办婚礼
    "unsafe_wedding_in_taimiao": [
        "新人在太庙大殿前举行婚礼",
    ],
    # 安全：普通正常场景（用来对冲，区分开）
    "safe_general_normal": [
        "正常的自然风光与城市建筑景观",
        "普通日常人文生活纪实画面",
        "自然山水、园林湖泊、户外原野常规景色",
        "城市街区、民用建筑、休闲公共场所普通场景",
        "大众旅游风光、自然生态、市井烟火普通场景",
        "常规人文景观、自然地貌、城市风貌的标准照片",
        "普通户外风景、室内场景、生活随拍正常画面",
        "开阔自然景致与城市人居的常规画面",
        "日常休闲、生活纪实、自然百态普通场景",
        "城乡风貌、自然景致、人文日常的通用照片",
    ],
}


# ========================== 工具函数区 ==========================

def apply_template_to_prompts(prompts, template):
    """为提示词列表统一应用模板"""
    return [template.format(p) for p in prompts]


def normalize_features(features):
    """对特征向量进行 L2 归一化"""
    return features / features.norm(p=2, dim=-1, keepdim=True)


def extract_text_features(texts, model, processor):
    """提取文本特征并归一化"""
    inputs = processor(text=texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        features = model.get_text_features(**inputs)
    if hasattr(features, "pooler_output"):
        features = features.pooler_output
    return normalize_features(features)


def extract_image_features(image, model, processor):
    """提取图像特征并归一化"""
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    if hasattr(features, "pooler_output"):
        features = features.pooler_output
    return normalize_features(features)


def classify_image(image_path, class_prompts, model, processor, template=CLIP_TEMPLATE):
    """
    完整的图像分类流程：提取特征 → 计算相似度 → logit_scale缩放 → softmax归一化
    :return: {类别名: 概率值(0~1)，所有类别概率之和为1}
    """
    # 1. 加载图像
    image = Image.open(image_path)

    # 2. 提取图像特征
    image_feat = extract_image_features(image, model, processor)  # (1, D)

    # 3. 获取模型学习到的 logit_scale（关键！CLIP训练时学到的温度参数）
    logit_scale = model.logit_scale.exp()  # exp(logit_scale)，典型值 ~100

    # 4. 计算每个类别的聚合相似度（相似度平均，而非特征平均）
    class_names = list(class_prompts.keys())
    aggregated_scores = []

    for class_name in class_names:
        raw_prompts = class_prompts[class_name]
        standardized = apply_template_to_prompts(raw_prompts, template)
        text_feat = extract_text_features(standardized, model, processor)  # (num_prompts, D)

        # 每个提示与图像独立计算相似度
        sims = (image_feat @ text_feat.T).squeeze(0)  # (num_prompts,)

        # 聚合：取所有提示相似度的均值（每个提示独立投票）
        aggregated_scores.append(sims.mean().item())

    # 5. 将聚合分数转为 tensor，乘以 logit_scale，再 softmax
    scores_tensor = torch.tensor(
        [aggregated_scores], device=image_feat.device
    )  # (1, num_classes)

    # 核心：乘以 logit_scale 缩放（与 CLIP 原始计算方式一致）
    scaled_scores = scores_tensor * logit_scale

    # softmax 转概率分布
    probs = scaled_scores.softmax(dim=1).squeeze(0)  # (num_classes,)

    # 6. 返回 {类别名: 概率}
    return {name: probs[i].item() for i, name in enumerate(class_names)}


def print_results(results):
    """格式化打印分类结果"""
    print("--- 分类概率结果 ---")
    for class_name, prob in results.items():
        bar = "█" * int(prob * 50)
        print(f"  类别: {class_name:40s} | 概率: {prob * 100:7.4f}%  {bar}")

    # 校验：概率之和应为 1.0
    total = sum(results.values())
    print(f"  {'概率合计':40s} | {total * 100:7.4f}%")

    top_class = max(results, key=results.get)
    top_prob = results[top_class]
    print(f"\n最佳匹配: {top_class} (置信度: {top_prob * 100:.2f}%)")


# ========================== 主程序区 ==========================

def main():
    # 加载模型
    print("正在加载模型...")
    model = ChineseCLIPModel.from_pretrained(MODEL_PATH)
    processor = ChineseCLIPProcessor.from_pretrained(MODEL_PATH)

    # 执行分类
    results = classify_image(IMAGE_PATH, CLASS_PROMPTS, model, processor)

    # 打印结果
    print_results(results)


if __name__ == "__main__":
    main()

