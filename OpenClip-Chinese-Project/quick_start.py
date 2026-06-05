from PIL import Image
import requests
from transformers import ChineseCLIPProcessor, ChineseCLIPModel

model = ChineseCLIPModel.from_pretrained("/root/autodl-tmp/chinese-clip-vit-base-patch16")
processor = ChineseCLIPProcessor.from_pretrained("./autodl-tmp/chinese-clip-vit-base-patch16")

images = "./image6.jpg"
image = Image.open(images)
# Squirtle, Bulbasaur, Charmander, Pikachu in English
texts = ["故宫里停放的奔驰", "太庙举行的婚礼", "苏州园林", "故宫"]

# compute image feature
inputs = processor(images=image, return_tensors="pt")
image_features = model.get_image_features(**inputs)
# 核心修改：如果返回的是对象，则取其 pooler_output 属性
if hasattr(image_features, "pooler_output"):
    image_features = image_features.pooler_output
else:
    image_features = image_features  # 如果已经是 tensor 则直接赋值

image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)  # normalize

# compute text features
inputs = processor(text=texts, padding=True, return_tensors="pt")
text_features = model.get_text_features(**inputs)
if hasattr(text_features, "pooler_output"):
    text_features = text_features.pooler_output
else:
    text_features = text_features

text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)  # normalize

# compute image-text similarity scores
inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)  # probs: [[1.2686e-03, 5.4499e-02, 6.7968e-04, 9.4355e-01]]
prob_list = probs[0].detach().cpu().numpy()

# 2. 循环打印结果
print("--- 分类概率结果 ---")
for label, prob in zip(texts, prob_list):
    print(f"标签: {label:10} | 概率: {prob * 100:7.4f}%")

# 3. 自动输出最高概率的项
top_index = prob_list.argmax()
print(f"\n最佳匹配: {texts[top_index]} (置信度: {prob_list[top_index] * 100:.2f}%)")