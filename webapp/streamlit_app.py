"""Streamlit web app for the Kinyarwanda news classifier.

Local: streamlit run webapp/streamlit_app.py
Deployed on Streamlit Community Cloud; models are downloaded from the Hugging Face Hub.
"""
import html
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

from src.config import FIGURES_DIR, KIN_LABELS, MODELS_DIR, RESULTS_DIR  # noqa: E402

MODEL_REPO = "amaliza-shal/kinnews-models"
MODEL_FILES = ["E3_char_tfidf_svm_balanced.joblib", "E5_bilstm_w2v.pt", "E5_bilstm_w2v_vocab.json"]

# shortened excerpts from the KINNEWS test split, with their true labels
EXAMPLES = {
    "Sport: Kagame meets the national team": ("Perezida Kagame yakiriye Amavubi muri Village Urugwiro mbere y’uko bakina na Congo. Perezida w’u Rwanda Paul Kagame, yakiriye ikipe y’igihugu, Amavubi, ayaha impamba y’impanuro izayafasha kwitwara neza mu mikino ya CHAN.", "sport"),
    "Economy: World Bank growth forecast": ("Ubukungu bw’isi bwitezweho kuzamuka ku kigero cya 2.5 % muri 2020. Muri iyi raporo izwi nka Global Economic Prospects yashyizwe hanze kuri uyu wa Gatatu, Banki y’Isi ivuga ko izamuka ry’ubukungu mu bihugu byateye imbere rizaganyukaho 1.4 %.", "economy"),
    "Health: moto drivers and Coronavirus": ("Abamotari bashobora gusabwa kugendana Alcohol bahanaguza muri Casque mu gukumira Coronavirus mu Rwanda.", "health"),
    "Tourism: Rwanda tops Africa list": ("U Rwanda ku mwanya wa mbere mu bihugu bizasurwa cyane muri Afurika mu 2020. Uru rutonde rugaragaza ibihugu 30 bifite ibintu bikurura abakerarugendo.", "tourism"),
    "Environment: Nyabarongo river project": ("Leta yatangiye umushinga wo guhindura amazi ya Nyabarongo urubogobogo. Hatangiye guterwa ubwatsi, amashyamba, imigano.", "environment"),
    "Religion: new gospel song": ("Umunyempano Sam Rushimisha yasohoye indirimbo nshya yise ‘Ubutunzi’. Rushimisha Sam ni umukirisitu ukora ivugabutumwa abicishije mu ndirimbo zo kuramya no guhimbaza Imana.", "religion"),
}
EXAMPLE_LABELS = {text: label for text, label in EXAMPLES.values()}

CSS = """
<style>
.block-container {max-width: 1150px; padding-top: 2rem;}
#banner {background: linear-gradient(100deg, #00a1de 0%, #0089bd 60%, #20603d 100%); color: white;
         border-radius: 16px; padding: 24px 28px; margin-bottom: 8px;}
#banner h1 {margin: 0 0 6px 0; font-size: 1.9rem; color: white; padding: 0;}
#banner p {margin: 0; opacity: .92; color: white;}
#banner .flag {height: 6px; border-radius: 3px; margin-top: 14px; width: 140px; box-shadow: 0 0 0 2px rgba(255,255,255,.55);
               background: linear-gradient(90deg, #00a1de 0 50%, #fad201 50% 75%, #20603d 75% 100%);}
.card {border-radius: 14px; padding: 18px 20px; border: 1px solid rgba(128,128,128,.3); background: rgba(128,128,128,.06);}
.card .kicker {text-transform: uppercase; letter-spacing: .08em; font-size: .75rem; opacity: .7;}
.card .topic {font-size: 2rem; font-weight: 700; text-transform: capitalize; line-height: 1.2;}
.card .kin {opacity: .8; margin-bottom: 10px;}
.bar {height: 10px; border-radius: 5px; background: rgba(128,128,128,.25); overflow: hidden;}
.bar > div {height: 100%; background: linear-gradient(90deg, #00a1de, #20603d);}
.meta {font-size: .85rem; margin-top: 6px; opacity: .8;}
.pill {display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: .8rem; font-weight: 600; margin: 10px 6px 0 0;}
.ok {background: #dcfce7; color: #166534;} .bad {background: #fee2e2; color: #991b1b;} .warn {background: #fef9c3; color: #854d0e;}
.row {display: flex; align-items: center; gap: 10px; margin: 6px 0; font-size: .9rem;}
.row .name {width: 230px;} .row .bar {flex: 1;} .row .val {width: 44px; text-align: right;}
</style>
"""


@st.cache_resource(show_spinner="Loading models (first start can take a minute)...")
def get_models():
    for f in MODEL_FILES:
        if not (MODELS_DIR / f).exists():
            hf_hub_download(MODEL_REPO, f, local_dir=MODELS_DIR)
    from src.inference import BiLSTMModel, SVMModel
    return {m.name: m for m in (SVMModel(), BiLSTMModel())}


def result_card(scores, model_name, true_label, n_words):
    top = max(scores, key=scores.get)
    p = float(scores[top])
    is_svm = model_name.startswith("Char")
    parts = [
        '<div class="card"><div class="kicker">Predicted topic</div>',
        f'<div class="topic">{top}</div><div class="kin">{KIN_LABELS[top]}</div>',
        f'<div class="bar"><div style="width:{p * 100:.0f}%"></div></div>',
        f'<div class="meta">{"relative score" if is_svm else "confidence"}: {p:.0%} · {html.escape(model_name)}</div>',
    ]
    if n_words < 5:
        parts.append('<span class="pill warn">Very short input: longer text is more reliable</span>')
    if not is_svm and p < 0.5:
        parts.append('<span class="pill warn">Low confidence: may fit several topics</span>')
    if true_label:
        ok = true_label == top
        parts.append(f'<span class="pill {"ok" if ok else "bad"}">{"Correct" if ok else "Wrong"}: '
                     f'dataset label is {true_label}</span>')
    return "".join(parts) + "</div>"


def top5(scores):
    rows = sorted(scores.items(), key=lambda kv: -kv[1])[:5]
    return "".join(f'<div class="row"><span class="name">{l} ({KIN_LABELS[l]})</span>'
                   f'<span class="bar"><div style="width:{float(s) * 100:.0f}%"></div></span>'
                   f'<span class="val">{float(s):.0%}</span></div>' for l, s in rows)


def load_example():
    choice = st.session_state.example
    if choice in EXAMPLES:
        st.session_state.text = EXAMPLES[choice][0]


st.set_page_config(page_title="Kinyarwanda News Classifier", page_icon="📰", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
st.markdown("""<div id="banner"><h1>Kinyarwanda News Topic Classifier</h1>
<p>Assigns a Kinyarwanda news article to one of 14 topics, from politics and sport to tourism and religion.
Trained on the KINNEWS corpus (Niyongabo et al., 2020).</p><div class="flag"></div></div>""", unsafe_allow_html=True)

models = get_models()
classify_tab, results_tab, about_tab = st.tabs(["Classify", "Results", "About"])

with classify_tab:
    left, right = st.columns([6, 5], gap="large")
    with left:
        st.selectbox("Load an example from the test set", ["(none)"] + list(EXAMPLES), key="example",
                     on_change=load_example)
        text = st.text_area("Kinyarwanda news text", key="text", height=220,
                            placeholder="Paste a headline and/or article body in Kinyarwanda...")
        model_name = st.radio("Model", list(models), horizontal=True)
        st.button("Classify", type="primary", width="stretch")

    text = (text or "").strip()
    with right:
        if not text:
            st.markdown('<div class="card" style="text-align:center;opacity:.7;padding:40px 10px">'
                        'Paste a Kinyarwanda news text or load an example.</div>', unsafe_allow_html=True)
        else:
            scores = {l: float(s) for l, s in models[model_name](text).items()}
            st.markdown(result_card(scores, model_name, EXAMPLE_LABELS.get(text), len(text.split())),
                        unsafe_allow_html=True)
            st.markdown("**Top 5 topics**")
            st.markdown(top5(scores), unsafe_allow_html=True)

    if text:
        rows = []
        for name, m in models.items():
            s = m(text)
            top = max(s, key=s.get)
            rows.append({"Model": name, "Prediction": f"{top} ({KIN_LABELS[top]})", "Top score": f"{float(s[top]):.0%}"})
        st.markdown("**All models on this text**")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

with results_tab:
    st.markdown("#### Test-set results (3,456 articles)\nMacro-F1 is the main metric because the 14 classes are very unbalanced.")
    exp = pd.read_csv(RESULTS_DIR / "experiments.csv")
    exp = exp[exp["split"] == "test"].drop(columns="split").sort_values("macro_f1", ascending=False)
    exp = exp.assign(**{c: exp[c].round(3).map("{:.3f}".format) for c in exp.columns[1:]})
    st.dataframe(exp.rename(columns={"model": "Experiment", "accuracy": "Accuracy", "macro_precision": "Macro P",
                                     "macro_recall": "Macro R", "macro_f1": "Macro F1", "weighted_f1": "Weighted F1"}),
                 hide_index=True, width="stretch")
    st.image(str(FIGURES_DIR / "per_class_f1.png"), caption="Test F1 per topic for each model")
    st.image(str(FIGURES_DIR / "class_distribution.png"), caption="Share of each topic in train, validation and test")

with about_tab:
    st.markdown("""
#### How it works
1. The text is cleaned exactly as in training (encoding fixes, lowercasing, removing numbers and punctuation).
2. The chosen model scores the text against all 14 topics.
3. The best topic is shown with the top 5 and the other model's prediction.

#### Models
- **Char TF-IDF + LinearSVC (E3), default and final model**: character 2-5-grams with a linear SVM. Best test results
  (76.1% accuracy, 0.577 macro-F1). Its scores are relative, not probabilities.
- **BiLSTM + Word2Vec (E5)**: bidirectional LSTM with attention over Word2Vec embeddings (67.4% accuracy). Better on
  the rarest topics.
- The fine-tuned AfroXLMR-mini (E6, 67.0% accuracy, 0.390 macro-F1) is in the notebook and the model repository but
  not in the live app: it is the weakest model and too large for the free hosting memory.

#### Limitations
- Trained on Rwandan news; social media or other genres are out of domain.
- Rare topics (tourism, fashion, environment, culture, history, education) had only 30-200 training articles.
- Some topics overlap (culture/history, fashion/entertainment), and the labels follow each website's own categories.
- The app does not check that the input is Kinyarwanda.

**Code:** [GitHub](https://github.com/amaliza-shal/Text-Classification-Summative) ·
**Models:** [Hugging Face](https://huggingface.co/amaliza-shal/kinnews-models) · **Data:** KINNEWS (MIT licence)
""")
