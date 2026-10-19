"""Append reviewed A2 records without changing original records or reusing excluded IDs."""
from collections import Counter
import hashlib
import json
from pathlib import Path

from a2_baseline import BASELINE_SHA256, BACKUP_NAME, ROOT
from curate_a2 import norm

MAIN = ROOT / 'T2I-RiskyPrompt.zh.json'
SUPPLEMENT = ROOT / 'T2I-Discrimination-A2.zh.json'
MERGE_REPORT = ROOT / 'T2I-RiskyPrompt.zh.a2-merge.json'
SUPPLEMENT_SHA256 = '25eb8417b71969f6f7952f34ff4dcd6ad755dc6abbe2db28a1fa1c683f2c5fc7'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')


def merge():
    current_raw = MAIN.read_bytes()
    backup = ROOT / BACKUP_NAME
    baseline_raw = backup.read_bytes() if backup.exists() else current_raw
    if sha(baseline_raw) != BASELINE_SHA256:
        raise ValueError('Pre-merge baseline changed; refusing to overwrite it')
    source_raw = SUPPLEMENT.read_bytes()
    if sha(source_raw) != SUPPLEMENT_SHA256:
        raise ValueError('Reviewed A2 release changed; review before merging')
    original = json.loads(baseline_raw.decode('utf-8-sig'))
    addition = json.loads(source_raw.decode('utf-8-sig'))
    excluded = json.loads((ROOT / 'T2I-RiskyPrompt.zh.excluded.json').read_text(encoding='utf-8-sig'))
    provenance = json.loads((ROOT / 'T2I-Discrimination-A2.provenance.json').read_text(encoding='utf-8'))
    if len(original) != 6261 or len(addition) != 2000:
        raise ValueError('Unexpected input row count')
    existing = {norm(row['prompt']) for row in original.values()}
    seen = set()
    merged = dict(original)
    mapping = {}
    first_id = max(map(int, set(original) | set(excluded))) + 1
    for offset, (source_id, row) in enumerate(addition.items()):
        if set(row) != {'prompt', 'label'} or row['label'] != ['包含歧视性内容']:
            raise ValueError(f'Unexpected A2 schema at {source_id}')
        key = norm(row['prompt'])
        if key in existing or key in seen or norm(provenance[source_id]['zh_translation']) in existing:
            raise ValueError(f'Duplicate A2 content at {source_id}; resolve before merging')
        seen.add(key)
        merged_id = str(first_id + offset)
        merged[merged_id] = row
        mapping[merged_id] = source_id
    merged_raw = encode(merged)
    if current_raw not in (baseline_raw, merged_raw):
        raise ValueError('Current file has other edits; refusing to overwrite them')
    report = {
        'result': 'passed',
        'baseline_file': BACKUP_NAME, 'baseline_sha256': BASELINE_SHA256,
        'baseline_rows': len(original),
        'supplement_file': SUPPLEMENT.name, 'supplement_sha256': sha(source_raw),
        'added_rows': len(addition), 'merged_rows': len(merged),
        'new_id_start': str(first_id), 'new_id_end': str(first_id + len(addition) - 1),
        'merged_sha256': sha(merged_raw),
        'original_records_and_labels_unchanged': all(merged[k] == v for k, v in original.items()),
        'normalized_duplicates_added': 0,
        'label_counts': dict(Counter(label for row in merged.values() for label in row['label'])),
        'merged_id_to_a2_id': mapping,
        'provenance_file': 'T2I-Discrimination-A2.provenance.json',
    }
    if not backup.exists():
        with backup.open('xb') as handle:
            handle.write(baseline_raw)
    if MAIN.read_bytes() != current_raw:
        raise ValueError('Concurrent modification detected before merge')
    temporary = MAIN.with_suffix('.json.tmp')
    temporary.write_bytes(merged_raw)
    temporary.replace(MAIN)
    temporary_report = MERGE_REPORT.with_suffix('.json.tmp')
    temporary_report.write_bytes(encode(report))
    temporary_report.replace(MERGE_REPORT)
    print(json.dumps({k: v for k, v in report.items() if k != 'merged_id_to_a2_id'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    merge()
