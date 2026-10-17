"""Clean KINNEWS, remove duplicates/leakage and write train/val/test splits.

Run: python -m src.preprocess
"""
import json
import re
import unicodedata

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATA_PROCESSED, DATA_RAW, LABELS, RESULTS_DIR, SEED, VAL_FRACTION

# Some articles were saved as UTF-8 but read back as latin-1 (e.g. "â\x80\x99" for "’")
MOJIBAKE = re.compile(r"[\xc2-\xf4][\x80-\xbf]{1,3}")
URL = re.compile(r"https?://\S+|www\.\S+")
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
NON_LETTER = re.compile(r"[^a-z]+")


def fix_mojibake(text):
    def repair(m):
        try:
            return m.group().encode("latin-1").decode("utf-8")
        except UnicodeError:
            return m.group()
    return MOJIBAKE.sub(repair, text)


def light_clean(text):
    # input for the transformer: keep case and punctuation, the tokenizer was trained with them
    text = unicodedata.normalize("NFKC", fix_mojibake(str(text)))
    text = URL.sub(" ", text)
    return " ".join(text.split())


def clean_text(text):
    # input for TF-IDF / Word2Vec models. Apostrophes become spaces, so the elided
    # forms "n'abana", "y'u Rwanda" give the same stem token as "abana", "u"
    text = light_clean(text).lower()
    text = NUMBER.sub(" ", text)
    return NON_LETTER.sub(" ", text).strip()


def load(split):
    df = pd.read_parquet(DATA_RAW / f"kinnews_{split}.parquet")
    df["label"] = df["label"].map(lambda i: LABELS[i])
    raw = df["title"].fillna("") + " . " + df["content"].fillna("")
    df["text"] = raw.map(clean_text)
    df["text_tf"] = raw.map(light_clean)
    return df


def dedup(df):
    n = len(df)
    df = df[df["text"].str.split().str.len() >= 10]
    n_short = n - len(df)
    # the same article sometimes appears under two categories; we can't tell which is right
    conflicting = df.groupby("text")["label"].transform("nunique") > 1
    n_conflict = int(conflicting.sum())
    df = df[~conflicting].drop_duplicates("text")
    stats = {"rows": n, "too_short": n_short, "conflicting_label_rows": n_conflict,
             "duplicate_rows": n - n_short - n_conflict - len(df), "unique": len(df)}
    return df, stats


def main():
    train, train_stats = dedup(load("train"))
    test, test_stats = dedup(load("test"))

    leak = test["text"].isin(set(train["text"]))
    test_stats["also_in_train"] = int(leak.sum())
    test = test[~leak]

    train, val = train_test_split(train, test_size=VAL_FRACTION, stratify=train["label"],
                                  random_state=SEED)

    splits = {"train": train, "val": val, "test": test}
    for name, df in splits.items():
        df[["label", "title", "text", "text_tf", "url"]].to_csv(DATA_PROCESSED / f"{name}.csv", index=False)

    summary = {
        "train_raw": train_stats,
        "test_raw": test_stats,
        "sizes": {k: len(v) for k, v in splits.items()},
        "class_counts": {k: v["label"].value_counts().reindex(LABELS, fill_value=0).to_dict()
                         for k, v in splits.items()},
    }
    (RESULTS_DIR / "data_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
