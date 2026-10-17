"""Upload the Gradio app, the trained models and the result figures to a Hugging Face Space.

Needs `hf auth login` with a write token, and the trained models in models/.
Run: python deploy_space.py --space <username>/kinnews-classifier
"""
import argparse
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

SPACE_README = """---
title: Kinyarwanda News Classifier
emoji: 📰
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.30.0
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
---
Topic classifier for Kinyarwanda news (14 KINNEWS categories).
Code: https://github.com/amaliza-shal/Text-Classification-Summative
"""

# CPU-only torch keeps the Space build small
SPACE_REQUIREMENTS = """--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.14.1
transformers==5.19.0
scikit-learn==1.9.1
numpy==2.4.6
pandas==3.0.6
joblib==1.6.0
gensim==4.4.0
matplotlib==3.11.2
seaborn==0.13.2
"""

FILES = [
    "app.py",
    "src/__init__.py", "src/config.py", "src/preprocess.py", "src/bilstm.py", "src/evaluate.py", "src/inference.py",
    "models/E3_char_tfidf_svm_balanced.joblib", "models/E5_bilstm_w2v.pt", "models/E5_bilstm_w2v_vocab.json",
    "results/experiments.csv", "results/figures/per_class_f1.png", "results/figures/class_distribution.png",
]

ap = argparse.ArgumentParser()
ap.add_argument("--space", required=True)
args = ap.parse_args()

api = HfApi()
api.create_repo(args.space, repo_type="space", space_sdk="gradio", exist_ok=True)

with tempfile.TemporaryDirectory() as tmp:
    Path(tmp, "README.md").write_text(SPACE_README, encoding="utf-8")
    Path(tmp, "requirements.txt").write_text(SPACE_REQUIREMENTS)
    api.upload_folder(folder_path=tmp, repo_id=args.space, repo_type="space")

for path in FILES:
    api.upload_file(path_or_fileobj=path, path_in_repo=path, repo_id=args.space, repo_type="space")
api.upload_folder(folder_path="models/E6_afroxlmr_mini", path_in_repo="models/E6_afroxlmr_mini",
                  repo_id=args.space, repo_type="space", ignore_patterns=["checkpoints/*"])

print(f"https://huggingface.co/spaces/{args.space}")
