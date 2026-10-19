"""Locate the immutable RiskyPrompt baseline before the authorized A2 merge."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent / 'zh_data'
BASELINE_SHA256 = '4fd4a7f38577c8d3aa146ef8da33721337ff26a7c36f2758d8db828f9af9eee4'
BACKUP_NAME = 'T2I-RiskyPrompt.zh.before-A2.json'


def baseline_path():
    backup = ROOT / BACKUP_NAME
    path = backup if backup.exists() else ROOT / 'T2I-RiskyPrompt.zh.json'
    if hashlib.sha256(path.read_bytes()).hexdigest() != BASELINE_SHA256:
        raise ValueError('RiskyPrompt pre-merge baseline checksum mismatch')
    return path
