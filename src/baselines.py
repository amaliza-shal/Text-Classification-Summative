"""Classical baselines on TF-IDF features. Run: python -m src.baselines

E0 majority class           - floor every model must beat
E1 TF-IDF words + LogReg    - main baseline
E2 E1 + class_weight        - does reweighting help the rare classes?
E3 TF-IDF char n-grams + LinearSVC - sub-word features for an agglutinative language
"""
import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from src.config import DATA_PROCESSED, MODELS_DIR, SEED
from src.evaluate import evaluate

train, val, test = (pd.read_csv(DATA_PROCESSED / f"{s}.csv") for s in ("train", "val", "test"))

word_tfidf = dict(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True, max_features=200_000)

models = {
    "E0_majority": DummyClassifier(strategy="most_frequent"),
    "E1_tfidf_logreg": make_pipeline(
        TfidfVectorizer(**word_tfidf), LogisticRegression(C=10, max_iter=2000)),
    "E2_tfidf_logreg_balanced": make_pipeline(
        TfidfVectorizer(**word_tfidf), LogisticRegression(C=10, max_iter=2000, class_weight="balanced")),
    "E3_char_tfidf_svm_balanced": make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=300_000),
        LinearSVC(C=0.5, class_weight="balanced", random_state=SEED)),
}

for name, model in models.items():
    model.fit(train["text"], train["label"])
    evaluate(name, val, model.predict(val["text"]), split="val")
    evaluate(name, test, model.predict(test["text"]), split="test")
    if name != "E0_majority":
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")
