"""Metrics, confusion matrices and prediction dumps shared by every model."""
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)

from src.config import FIGURES_DIR, LABELS, RESULTS_DIR


def evaluate(name, df, preds, split="test"):
    """preds are label strings in the same order as df."""
    y = df["label"].tolist()
    p, r, f1, _ = precision_recall_fscore_support(y, preds, labels=LABELS, average="macro", zero_division=0)
    scores = {
        "model": name,
        "split": split,
        "accuracy": round(accuracy_score(y, preds), 4),
        "macro_precision": round(p, 4),
        "macro_recall": round(r, 4),
        "macro_f1": round(f1, 4),
        "weighted_f1": round(precision_recall_fscore_support(
            y, preds, labels=LABELS, average="weighted", zero_division=0)[2], 4),
    }
    print(json.dumps(scores))

    out = RESULTS_DIR / name
    out.mkdir(exist_ok=True)
    report = classification_report(y, preds, labels=LABELS, output_dict=True, zero_division=0)
    (out / f"{split}_report.json").write_text(json.dumps(report, indent=2))
    (out / f"{split}_scores.json").write_text(json.dumps(scores, indent=2))

    df.assign(pred=preds)[["label", "pred", "title", "text"]].to_csv(out / f"{split}_predictions.csv", index=False)

    cm = confusion_matrix(y, preds, labels=LABELS, normalize="true")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, xticklabels=LABELS, yticklabels=LABELS, cmap="Blues", annot=True, fmt=".2f",
                annot_kws={"size": 7}, cbar=False, ax=ax)
    ax.set(xlabel="predicted", ylabel="true", title=f"{name} ({split}) - row-normalised confusion matrix")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"cm_{name}_{split}.png", dpi=150)
    plt.close(fig)

    log = RESULTS_DIR / "experiments.csv"
    rows = pd.read_csv(log) if log.exists() else pd.DataFrame()
    if len(rows):
        rows = rows[~((rows["model"] == name) & (rows["split"] == split))]
    pd.concat([rows, pd.DataFrame([scores])]).to_csv(log, index=False)
    return scores
