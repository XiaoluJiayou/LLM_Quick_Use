"""图片生成输入在加载模型前完成规范化，避免同名文件覆盖。"""
from __future__ import annotations


def image_prompt_records(rows, base_seed=0):
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("BASE_SEED 必须是非负整数")
    result, seen = [], set()
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"第 {position} 行必须是对象")
        prompt = row.get("提示词")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"第 {position} 行提示词必须是非空文本")
        raw = row.get("ID", position)
        if isinstance(raw, bool) or not str(raw).isdecimal():
            raise ValueError(f"第 {position} 行 ID 必须是非负整数")
        record_id = int(raw)
        if record_id + base_seed >= 2 ** 64:
            raise ValueError(f"第 {position} 行 ID + BASE_SEED 超出随机种子范围")
        if record_id in seen:
            raise ValueError(f"规范化后 ID 重复: {record_id}；会覆盖同名图片")
        seen.add(record_id)
        result.append({"id": record_id, "prompt": prompt.strip()})
    if not result:
        raise ValueError("输入不能为空")
    return result
