"""Train skip-gram Word2Vec on the KINNEWS training text. Run: python -m src.word2vec"""
import pandas as pd
from gensim.models import Word2Vec

from src.config import DATA_PROCESSED, MODELS_DIR, SEED

sentences = [t.split() for t in pd.read_csv(DATA_PROCESSED / "train.csv")["text"]]

w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=2, sg=1, negative=10,
               epochs=10, workers=4, seed=SEED)
w2v.wv.save(str(MODELS_DIR / "w2v_kin_100d.kv"))
print("vocab:", len(w2v.wv))

# sanity check for the report: nearest neighbours of a few topic words
for w in ["umupira", "ubukungu", "indirimbo", "ubuzima", "kiliziya", "abanyeshuri", "urukundo"]:
    if w in w2v.wv:
        print(w, [n for n, _ in w2v.wv.most_similar(w, topn=6)])
