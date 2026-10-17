"""BiLSTM + attention pooling classifier.

python -m src.bilstm --emb random   # E4
python -m src.bilstm --emb w2v      # E5 (embeddings initialised from src/word2vec.py)
"""
import argparse
import json
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from gensim.models import KeyedVectors
from sklearn.metrics import f1_score

from src.config import DATA_PROCESSED, LABEL2ID, LABELS, MODELS_DIR, RESULTS_DIR, SEED
from src.evaluate import evaluate

MAX_LEN = 400  # covers the median document (~300 words); longer ones are truncated
PAD, UNK = 0, 1


class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size, emb_dim=100, hidden=128, n_classes=len(LABELS), dropout=0.4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.att = nn.Linear(2 * hidden, 1)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(2 * hidden, n_classes)

    def forward(self, x):
        mask = x != PAD
        h, _ = self.lstm(self.drop(self.emb(x)))           # (B, T, 2H)
        scores = self.att(h).squeeze(-1).masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1)             # attention over time steps
        doc = (weights.unsqueeze(-1) * h).sum(1)           # weighted sum of hidden states
        return self.out(self.drop(doc))


def build_vocab(texts, min_count=2):
    counts = Counter(w for t in texts for w in t.split())
    words = [w for w, c in counts.items() if c >= min_count]
    return {w: i + 2 for i, w in enumerate(words)}


def encode(texts, vocab):
    x = np.zeros((len(texts), MAX_LEN), dtype=np.int64)
    for i, t in enumerate(texts):
        ids = [vocab.get(w, UNK) for w in t.split()[:MAX_LEN]]
        x[i, :len(ids)] = ids
    return torch.from_numpy(x)


def predict(model, x, batch_size=128):
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(x[i:i + batch_size]) for i in range(0, len(x), batch_size)])
    return logits.argmax(1).numpy()


def train(emb="w2v", epochs=12, lr=2e-3, batch_size=32):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    name = "E5_bilstm_w2v" if emb == "w2v" else "E4_bilstm_random"

    train, val, test = (pd.read_csv(DATA_PROCESSED / f"{s}.csv") for s in ("train", "val", "test"))
    vocab = build_vocab(train["text"])
    x_train, x_val, x_test = (encode(d["text"], vocab) for d in (train, val, test))
    y_train = torch.tensor(train["label"].map(LABEL2ID).values)

    model = BiLSTMAttention(len(vocab) + 2)
    if emb == "w2v":
        kv = KeyedVectors.load(str(MODELS_DIR / "w2v_kin_100d.kv"))
        weights = model.emb.weight.data
        hits = 0
        for w, i in vocab.items():
            if w in kv:
                weights[i] = torch.from_numpy(kv[w].copy())
                hits += 1
        print(f"initialised {hits}/{len(vocab)} embeddings from word2vec")

    # inverse-frequency class weights, same idea as class_weight="balanced" in E2/E3
    counts = np.bincount(y_train.numpy(), minlength=len(LABELS))
    class_w = torch.tensor(len(y_train) / (len(LABELS) * np.maximum(counts, 1)), dtype=torch.float)
    loss_fn = nn.CrossEntropyLoss(weight=class_w)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_f1, history = -1, []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(x_train))
        total = 0
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            logits = model(x_train[idx])
            loss = loss_fn(logits, y_train[idx])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(idx)

        val_pred = [LABELS[i] for i in predict(model, x_val)]
        val_f1 = f1_score(val["label"], val_pred, labels=LABELS, average="macro", zero_division=0)
        history.append({"epoch": epoch + 1, "train_loss": round(total / len(perm), 4), "val_macro_f1": round(val_f1, 4)})
        print(history[-1])
        if val_f1 > best_f1:  # keep the checkpoint with the best validation macro-F1
            best_f1 = val_f1
            torch.save(model.state_dict(), MODELS_DIR / f"{name}.pt")

    model.load_state_dict(torch.load(MODELS_DIR / f"{name}.pt"))
    (MODELS_DIR / f"{name}_vocab.json").write_text(json.dumps(vocab))
    evaluate(name, val, [LABELS[i] for i in predict(model, x_val)], split="val")
    evaluate(name, test, [LABELS[i] for i in predict(model, x_test)], split="test")
    pd.DataFrame(history).to_csv(RESULTS_DIR / f"{name}_history.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", choices=["random", "w2v"], default="w2v")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch_size", type=int, default=32)
    train(**vars(ap.parse_args()))
