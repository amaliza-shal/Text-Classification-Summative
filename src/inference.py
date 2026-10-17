"""Load the trained models and score new text. Used by app.py and the notebook."""
import json

import joblib
import numpy as np
import torch

from src.bilstm import BiLSTMAttention, encode
from src.config import LABELS, MODELS_DIR
from src.preprocess import clean_text, light_clean

TRANSFORMER_DIR = MODELS_DIR / "E6_afroxlmr_mini"


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


class SVMModel:
    name = "Char TF-IDF + LinearSVC (E3)"

    def __init__(self):
        self.pipe = joblib.load(MODELS_DIR / "E3_char_tfidf_svm_balanced.joblib")

    def __call__(self, text):
        # LinearSVC has no probabilities; a softmax over the margins only ranks the classes
        margins = self.pipe.decision_function([clean_text(text)])[0]
        return dict(zip(self.pipe.classes_, softmax(margins)))


class BiLSTMModel:
    name = "BiLSTM + Word2Vec (E5)"

    def __init__(self):
        self.vocab = json.loads((MODELS_DIR / "E5_bilstm_w2v_vocab.json").read_text())
        self.net = BiLSTMAttention(len(self.vocab) + 2)
        self.net.load_state_dict(torch.load(MODELS_DIR / "E5_bilstm_w2v.pt"))
        self.net.eval()

    def __call__(self, text):
        with torch.no_grad():
            logits = self.net(encode([clean_text(text)], self.vocab))[0].numpy()
        return dict(zip(LABELS, softmax(logits)))


class TransformerModel:
    name = "AfroXLMR-mini fine-tuned (E6)"

    def __init__(self, path=TRANSFORMER_DIR):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        if not (path / "config.json").exists():
            raise FileNotFoundError(path)
        self.tok = AutoTokenizer.from_pretrained(path)
        self.net = AutoModelForSequenceClassification.from_pretrained(path).eval()

    def __call__(self, text):
        x = self.tok(light_clean(text), truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            logits = self.net(**x).logits[0].numpy()
        return {self.net.config.id2label[i]: p for i, p in enumerate(softmax(logits))}


def load_models():
    """Returns {display name: model}, skipping models whose files are missing."""
    models = {}
    for cls in (SVMModel, TransformerModel, BiLSTMModel):  # best model first: the app uses it by default
        try:
            m = cls()
        except FileNotFoundError as e:
            print(f"skipping {cls.name}: {e}")
            continue
        models[m.name] = m
    return models
