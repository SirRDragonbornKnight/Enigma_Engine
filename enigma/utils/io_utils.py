from pathlib import Path
import json

def read_text(path):
    return Path(path).read_text(encoding="utf-8")

def write_text(path, txt):
    Path(path).write_text(txt, encoding="utf-8")

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")
