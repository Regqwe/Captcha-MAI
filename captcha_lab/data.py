"""Shared repository path and JSON output for the final model."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def save_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
