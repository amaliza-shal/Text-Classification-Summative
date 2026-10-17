"""Download the raw KINNEWS splits (Niyongabo et al., 2020; MIT licence) from the Hugging Face Hub.

Run: python -m src.download
"""
import urllib.request

from src.config import DATA_RAW

BASE = "https://huggingface.co/datasets/andreniyongabo/kinnews_kirnews/resolve/refs%2Fconvert%2Fparquet/kinnews_raw"

for split in ("train", "test"):
    dest = DATA_RAW / f"kinnews_{split}.parquet"
    if not dest.exists():
        urllib.request.urlretrieve(f"{BASE}/{split}/0000.parquet", dest)
    print(dest, dest.stat().st_size, "bytes")
