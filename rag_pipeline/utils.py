import os, hashlib

def mmss(seconds: float) -> str:
    seconds = max(0, float(seconds))
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"

def hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
