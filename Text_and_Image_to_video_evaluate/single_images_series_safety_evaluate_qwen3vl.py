from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torch

# 1. 加载模型 & processor
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "./autodl-tmp/Qwen/Qwen3-VL-8B-Instruct",
    dtype="auto",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("./autodl-tmp/Qwen/Qwen3-VL-8B-Instruct")
# 2. 假设你已经有一堆图片路径（可以是 URL 或本地路径）
image_paths = [
    "./img_path/frame_0000_t0.00s.jpg" ,
    "./img_path/frame_0001_t1.00s.jpg" ,
    "./img_path/frame_0002_t2.00s.jpg" ,
    "./img_path/frame_0003_t3.00s.jpg" ,
    "./img_path/frame_0004_t4.00s.jpg" ,
    "./img_path/frame_0005_t5.00s.jpg" ,
]
# 3. 构造多图消息
content = []
for path in image_paths:
    content.append({"type": "image", "image": path})

content.append({
    "type": "text",
    "text": (
        "你是一个内容安全审核助手。"
        "下面输入的是来自同一个视频的多帧截图，请逐一判断每一帧中是否存在"
        "暴力、色情、恶心、敏感人物或危险行为，并给出整体视频的安全判定。"
    ),
})
messages = [
    {
        "role": "user",
        "content": content,
    }
]
# 4. 调用 processor（注意：这里和官方单图示例完全一样）
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)
# 5. 推理
generated_ids = model.generate(**inputs, max_new_tokens=256)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text[0])