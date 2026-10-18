"""Upload the trained models to a Hugging Face model repository so the web app can download them.

Needs `hf auth login` with a write token. Run: python upload_models.py
"""
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

REPO = "amaliza-shal/kinnews-models"

CARD = """---
license: mit
language: rw
datasets: andreniyongabo/kinnews_kirnews
pipeline_tag: text-classification
---
# Kinyarwanda news topic classifiers (KINNEWS, deduplicated)

Models trained for https://github.com/amaliza-shal/Text-Classification-Summative.
Test set: 3,456 deduplicated KINNEWS articles, 14 topics.

| File | Model | Test accuracy | Test macro-F1 |
|---|---|---|---|
| `E3_char_tfidf_svm_balanced.joblib` | char 2-5-gram TF-IDF + LinearSVC (final model) | 0.761 | 0.577 |
| `E5_bilstm_w2v.pt` + `E5_bilstm_w2v_vocab.json` | BiLSTM + attention, Word2Vec init | 0.674 | 0.545 |
| `E6_afroxlmr_mini/` | AfroXLMR-mini fine-tuned (128 tokens) | 0.670 | 0.390 |

The scikit-learn model needs scikit-learn 1.9.1 and the text cleaning in `src/preprocess.py` of the GitHub repo.
"""

api = HfApi()
api.create_repo(REPO, repo_type="model", exist_ok=True)
with tempfile.TemporaryDirectory() as tmp:
    Path(tmp, "README.md").write_text(CARD, encoding="utf-8")
    api.upload_file(path_or_fileobj=str(Path(tmp, "README.md")), path_in_repo="README.md", repo_id=REPO)
api.upload_folder(folder_path="models", repo_id=REPO,
                  allow_patterns=["E3_char_tfidf_svm_balanced.joblib", "E5_bilstm_w2v.pt", "E5_bilstm_w2v_vocab.json",
                                  "E6_afroxlmr_mini/*"],
                  ignore_patterns=["*/checkpoints/*", "*/training_args.bin"])
print(f"https://huggingface.co/{REPO}")
