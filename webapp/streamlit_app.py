"""Streamlit web app for the Kinyarwanda news classifier.

Local: streamlit run webapp/streamlit_app.py
Deployed on Streamlit Community Cloud; the models are downloaded from the Hugging Face Hub.
"""
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

from src.config import KIN_LABELS, LABELS, MODELS_DIR, RESULTS_DIR  # noqa: E402

MODEL_REPO = "amaliza-shal/kinnews-models"
MODEL_FILES = ["E3_char_tfidf_svm_balanced.joblib", "E5_bilstm_w2v.pt", "E5_bilstm_w2v_vocab.json"]
GITHUB = "https://github.com/amaliza-shal/Text-Classification-Summative"

ICONS = {"politics": "🏛️", "sport": "⚽", "economy": "💰", "health": "🩺", "entertainment": "🎬", "history": "📜",
         "technology": "💻", "tourism": "🧳", "culture": "🥁", "fashion": "👗", "religion": "⛪",
         "environment": "🌿", "education": "🎓", "relationship": "💞"}

# shortened excerpts from the KINNEWS test split, with their true labels
EXAMPLES = {
    "⚽ Sport": ("Perezida Kagame yakiriye Amavubi muri Village Urugwiro mbere y’uko bakina na Congo. Perezida w’u Rwanda Paul Kagame, yakiriye ikipe y’igihugu, Amavubi, ayaha impamba y’impanuro izayafasha kwitwara neza mu mikino ya CHAN.", "sport"),
    "💰 Economy": ("Ubukungu bw’isi bwitezweho kuzamuka ku kigero cya 2.5 % muri 2020. Muri iyi raporo izwi nka Global Economic Prospects yashyizwe hanze kuri uyu wa Gatatu, Banki y’Isi ivuga ko izamuka ry’ubukungu mu bihugu byateye imbere rizaganyukaho 1.4 %.", "economy"),
    "🩺 Health": ("Abamotari bashobora gusabwa kugendana Alcohol bahanaguza muri Casque mu gukumira Coronavirus mu Rwanda.", "health"),
    "🧳 Tourism": ("U Rwanda ku mwanya wa mbere mu bihugu bizasurwa cyane muri Afurika mu 2020. Uru rutonde rugaragaza ibihugu 30 bifite ibintu bikurura abakerarugendo.", "tourism"),
    "🌿 Environment": ("Leta yatangiye umushinga wo guhindura amazi ya Nyabarongo urubogobogo. Hatangiye guterwa ubwatsi, amashyamba, imigano.", "environment"),
    "⛪ Religion": ("Umunyempano Sam Rushimisha yasohoye indirimbo nshya yise ‘Ubutunzi’. Rushimisha Sam ni umukirisitu ukora ivugabutumwa abicishije mu ndirimbo zo kuramya no guhimbaza Imana.", "religion"),
}
EXAMPLE_LABELS = {text: label for text, label in EXAMPLES.values()}

SERIES = {"Accuracy": "#2a78d6", "Macro-F1": "#eb6834"}  # validated categorical slots 1-2

CSS = """
<style>
.block-container {max-width: 1180px; padding-top: 1.6rem; padding-bottom: 3rem;}
#hero {position: relative; overflow: hidden; border-radius: 22px; padding: 34px 36px 30px; color: #fff;
       background: radial-gradient(1200px 400px at 85% -20%, rgba(250,210,1,.35), transparent 60%),
                   linear-gradient(115deg, #0077b6 0%, #00a1de 45%, #20603d 100%);
       box-shadow: 0 18px 40px -18px rgba(0,80,120,.55);}
#hero h1 {color: #fff; font-size: 2.35rem; line-height: 1.1; margin: 0; padding: 0; letter-spacing: -.02em;}
#hero .tag {display: inline-block; background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.35);
            padding: 4px 12px; border-radius: 999px; font-size: .8rem; margin-bottom: 14px; letter-spacing: .04em;}
#hero p {color: rgba(255,255,255,.92); font-size: 1.05rem; margin: 10px 0 0; max-width: 720px;}
.stats {display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 22px;}
.stat {background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.28); border-radius: 14px; padding: 12px 14px;
       backdrop-filter: blur(4px);}
.stat b {display: block; font-size: 1.55rem; color: #fff; line-height: 1.15;}
.stat span {font-size: .8rem; color: rgba(255,255,255,.85);}
@media (max-width: 760px) {.stats {grid-template-columns: repeat(2, 1fr);} #hero h1 {font-size: 1.8rem;}}

.panel {background: #fff; border: 1px solid #e6eaef; border-radius: 18px; padding: 20px 22px;
        box-shadow: 0 6px 20px -14px rgba(16,40,60,.35);}
.result .emoji {font-size: 2.6rem; line-height: 1;}
.result .kicker {text-transform: uppercase; letter-spacing: .09em; font-size: .72rem; color: #52514e; margin-top: 10px;}
.result .topic {font-size: 2.1rem; font-weight: 800; text-transform: capitalize; color: #0b0b0b; line-height: 1.15;}
.result .kin {color: #52514e; font-size: 1rem; margin-bottom: 14px;}
.empty {text-align: center; color: #52514e; padding: 46px 12px;}
.empty .emoji {font-size: 2.4rem; display: block; margin-bottom: 8px;}
.bar {height: 9px; border-radius: 6px; background: #edf0f3; overflow: hidden;}
.bar > div {height: 100%; border-radius: 6px; background: #2a78d6;}
.meta {font-size: .82rem; color: #52514e; margin-top: 7px;}
.pill {display: inline-block; padding: 4px 11px; border-radius: 999px; font-size: .78rem; font-weight: 600; margin: 12px 6px 0 0;}
.ok {background: #dcfce7; color: #166534;} .bad {background: #fee2e2; color: #991b1b;} .warn {background: #fef3c7; color: #854d0e;}
.row {display: grid; grid-template-columns: 26px 1fr 110px 44px; align-items: center; gap: 8px; margin: 7px 0; font-size: .9rem; color: #0b0b0b;}
.row .val {text-align: right; color: #52514e; font-variant-numeric: tabular-nums;}
.h {font-weight: 700; color: #0b0b0b; margin: 18px 0 4px;}
.compare {display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 6px;}
.mini {border: 1px solid #e6eaef; border-radius: 14px; padding: 12px 14px; background: #fafbfc;}
.mini .m {font-size: .78rem; color: #52514e;} .mini .p {font-size: 1.05rem; font-weight: 700; text-transform: capitalize; color: #0b0b0b;}
.cards {display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px;}
.card {border: 1px solid #e6eaef; border-radius: 16px; padding: 16px 18px; background: #fff;}
.card .t {font-weight: 700; color: #0b0b0b; margin-bottom: 4px;} .card .d {color: #52514e; font-size: .92rem;}
.tiles {display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 6px 0 14px;}
.tile {border: 1px solid #e6eaef; border-radius: 16px; padding: 14px 16px; background: #fff;}
.tile b {display: block; font-size: 1.7rem; color: #0b0b0b; line-height: 1.1;} .tile span {font-size: .82rem; color: #52514e;}
.steps {display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;}
.step {border-radius: 16px; padding: 16px; background: linear-gradient(160deg, #eef7fc, #fff); border: 1px solid #dbeaf3;}
.step .n {font-size: .75rem; font-weight: 700; color: #0077b6; letter-spacing: .08em;} .step .t {font-weight: 700; color: #0b0b0b; margin: 4px 0;}
.step .d {font-size: .88rem; color: #52514e;}
@media (max-width: 760px) {.cards, .compare {grid-template-columns: 1fr;} .tiles, .steps {grid-template-columns: repeat(2, 1fr);}}
.cards.three {grid-template-columns: repeat(3, 1fr);}
.topics {display: flex; flex-wrap: wrap; gap: 8px; margin-top: 6px;}
.topic-chip {background: #fff; border: 1px solid #e6eaef; border-radius: 999px; padding: 6px 12px; font-size: .9rem;
             color: #0b0b0b; text-transform: capitalize;}
.topic-chip i {color: #52514e; font-style: normal; text-transform: none; margin-left: 2px;}
@media (max-width: 760px) {.cards.three {grid-template-columns: 1fr;}}
.foot {text-align: center; color: #52514e; font-size: .85rem; margin-top: 30px;}
.foot a {color: #0077b6; text-decoration: none;}
div[data-testid="stTabs"] button p {font-size: 1rem; font-weight: 600;}
</style>
"""


@st.cache_resource(show_spinner="Loading the models (the first start takes about a minute)...")
def get_models():
    for f in MODEL_FILES:
        if not (MODELS_DIR / f).exists():
            hf_hub_download(MODEL_REPO, f, local_dir=MODELS_DIR)
    from src.inference import BiLSTMModel, SVMModel
    return {m.name: m for m in (SVMModel(), BiLSTMModel())}


@st.cache_data
def load_results():
    exp = pd.read_csv(RESULTS_DIR / "experiments.csv")
    test = exp[exp["split"] == "test"].set_index("model")
    per_class = pd.read_csv(RESULTS_DIR / "per_class_f1.csv", index_col=0)
    by_source = pd.read_csv(RESULTS_DIR / "accuracy_by_source.csv", index_col=0)
    data = json.loads((RESULTS_DIR / "data_summary.json").read_text())
    return test, per_class, by_source, data


def result_card(scores, model_name, true_label, n_words):
    top = max(scores, key=scores.get)
    p = scores[top]
    is_svm = model_name.startswith("Char")
    parts = [f'<div class="panel result"><div class="emoji">{ICONS[top]}</div><div class="kicker">Predicted topic</div>',
             f'<div class="topic">{top}</div><div class="kin">{KIN_LABELS[top]}</div>',
             f'<div class="bar"><div style="width:{p * 100:.0f}%"></div></div>',
             f'<div class="meta">{"Relative score" if is_svm else "Confidence"} {p:.0%} · {html.escape(model_name)}</div>']
    if true_label:
        ok = true_label == top
        parts.append(f'<span class="pill {"ok" if ok else "bad"}">{"✓ Correct" if ok else "✗ Wrong"} · '
                     f'dataset label: {true_label}</span>')
    if n_words < 5:
        parts.append('<span class="pill warn">⚠ Very short text</span>')
    elif not is_svm and p < 0.5:
        parts.append('<span class="pill warn">⚠ Low confidence</span>')
    rows = sorted(scores.items(), key=lambda kv: -kv[1])[:5]
    parts.append('<div class="h">Top 5 topics</div>')
    parts += [f'<div class="row"><span>{ICONS[l]}</span><span>{l} <span style="color:#52514e">· {KIN_LABELS[l]}</span></span>'
              f'<span class="bar"><div style="width:{s * 100:.0f}%"></div></span><span class="val">{s:.0%}</span></div>'
              for l, s in rows]
    return "".join(parts) + "</div>"


def compare_cards(text, models):
    cells = []
    for name, m in models.items():
        s = m(text)
        top = max(s, key=s.get)
        cells.append(f'<div class="mini"><div class="m">{html.escape(name)}</div>'
                     f'<div class="p">{ICONS[top]} {top}</div><div class="m">{float(s[top]):.0%}</div></div>')
    return '<div class="h">Both models on this text</div><div class="compare">' + "".join(cells) + "</div>"


def clear_example():
    st.session_state.example = None


def set_example():
    choice = st.session_state.example
    st.session_state.text = EXAMPLES[choice][0] if choice else ""


st.set_page_config(page_title="Kinyarwanda News Classifier", page_icon="📰", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
models = get_models()
test, per_class, by_source, data = load_results()
best = test.loc["E3_char_tfidf_svm_balanced"]

st.markdown(f"""
<div id="hero"><div class="tag">KINYARWANDA · NLP · TEXT CLASSIFICATION</div>
<h1>Kinyarwanda News Topic Classifier</h1>
<p>Paste a Kinyarwanda news article and get its topic, out of 14 categories, in a second.</p>
<div class="stats">
 <div class="stat"><b>14</b><span>news topics</span></div>
 <div class="stat"><b>{data["sizes"]["train"] + data["sizes"]["val"]:,}</b><span>unique training articles</span></div>
 <div class="stat"><b>{best["accuracy"]:.0%}</b><span>test accuracy</span></div>
 <div class="stat"><b>8</b><span>experiments compared</span></div>
</div></div>""", unsafe_allow_html=True)

tab_overview, tab_classify, tab_results, tab_data = st.tabs(["🏠 Overview", "🔍 Classify", "📊 Results", "🧹 The data"])

with tab_overview:
    st.markdown("""<div class="cards three">
     <div class="card"><div class="t">📰 The problem</div><div class="d">Rwandan news sites publish Kinyarwanda articles every
     day, but sorting them by topic is done by hand and almost no NLP tools exist for the language.</div></div>
     <div class="card"><div class="t">🌍 Why Kinyarwanda</div><div class="d">The project had to use an African language.
     Kinyarwanda is Rwanda's national language, has little labelled data, and has a real labelled news dataset (KINNEWS).</div></div>
     <div class="card"><div class="t">👥 Who it's for</div><div class="d">Newsrooms tagging articles, archives indexing
     Kinyarwanda content, and researchers who need topic labels.</div></div>
    </div>""", unsafe_allow_html=True)
    st.markdown('<div class="h">The 14 topics</div><div class="topics">' + "".join(
        f'<span class="topic-chip">{ICONS[l]} {l} <i>{KIN_LABELS[l]}</i></span>' for l in LABELS) + "</div>",
        unsafe_allow_html=True)
    st.markdown("""<div class="h">How it works</div><div class="steps">
     <div class="step"><div class="n">STEP 1</div><div class="t">Clean</div><div class="d">Remove duplicates, fix broken
     encoding, lowercase, drop numbers and punctuation.</div></div>
     <div class="step"><div class="n">STEP 2</div><div class="t">Represent</div><div class="d">Final model: character 2–5-grams
     weighted by TF-IDF, which match Kinyarwanda's many word forms.</div></div>
     <div class="step"><div class="n">STEP 3</div><div class="t">Score</div><div class="d">A linear SVM gives a score to each of
     the 14 topics.</div></div>
     <div class="step"><div class="n">STEP 4</div><div class="t">Predict</div><div class="d">The highest score is the topic;
     the top 5 show the alternatives.</div></div>
    </div>""", unsafe_allow_html=True)
    st.markdown("""<div class="h">Good to know</div><div class="cards three">
     <div class="card"><div class="d">Trained on Rwandan news, so social media and other genres are out of scope.</div></div>
     <div class="card"><div class="d">Rare topics (30–200 training articles) are less reliable than common ones.</div></div>
     <div class="card"><div class="d">Some topics overlap, such as culture/history and fashion/entertainment.</div></div>
    </div>""", unsafe_allow_html=True)

with tab_classify:
    st.pills("Try a real article from the test set", list(EXAMPLES), key="example", on_change=set_example)
    left, right = st.columns([11, 10], gap="large")
    with left:
        text = st.text_area("Kinyarwanda news text", key="text", height=210, on_change=clear_example,
                            placeholder="Paste a Kinyarwanda headline or article here...")
        model_name = st.radio("Model", list(models), horizontal=True,
                              help="E3 is the final model. E5 is shown for comparison.")
        st.button("Classify  →", type="primary", width="stretch")
    text = (text or "").strip()
    with right:
        if text:
            scores = {l: float(s) for l, s in models[model_name](text).items()}
            st.markdown(result_card(scores, model_name, EXAMPLE_LABELS.get(text), len(text.split())),
                        unsafe_allow_html=True)
            st.markdown(compare_cards(text, models), unsafe_allow_html=True)
        else:
            st.markdown('<div class="panel empty"><span class="emoji">📰</span>Pick an example or paste an article '
                        'to see its topic.</div>', unsafe_allow_html=True)

with tab_results:
    e1 = test.loc["E1_tfidf_logreg"]
    st.markdown(f"""<div class="tiles">
      <div class="tile"><b>{best["accuracy"]:.1%}</b><span>accuracy, final model (E3)</span></div>
      <div class="tile"><b>{best["macro_f1"]:.3f}</b><span>macro-F1, final model</span></div>
      <div class="tile"><b>{e1["macro_f1"]:.3f}</b><span>macro-F1, baseline (E1)</span></div>
      <div class="tile"><b>+{best["macro_f1"] - e1["macro_f1"]:.3f}</b><span>gain over the baseline</span></div>
    </div>""", unsafe_allow_html=True)

    names = {"E0_majority": "E0 Majority class", "E1_tfidf_logreg": "E1 Word TF-IDF + LogReg",
             "E2_tfidf_logreg_balanced": "E2 + class weights", "E3_char_tfidf_svm_balanced": "E3 Char TF-IDF + SVM",
             "E4_bilstm_random": "E4 BiLSTM", "E5_bilstm_w2v": "E5 BiLSTM + Word2Vec",
             "E6_afroxlmr_mini": "E6 AfroXLMR-mini"}
    long = (test[["accuracy", "macro_f1"]].rename(columns={"accuracy": "Accuracy", "macro_f1": "Macro-F1"})
            .rename(index=names).reset_index(names="Experiment").melt("Experiment", var_name="Metric", value_name="Score"))
    order = [names[m] for m in names]
    chart = alt.Chart(long).mark_bar(cornerRadiusEnd=4, height=11).encode(
        y=alt.Y("Experiment:N", sort=order, title=None, axis=alt.Axis(labelLimit=220)),
        yOffset=alt.YOffset("Metric:N", sort=list(SERIES)),
        x=alt.X("Score:Q", scale=alt.Scale(domain=[0, 1]), title="test score",
                axis=alt.Axis(format=".0%", grid=True, tickCount=5)),
        color=alt.Color("Metric:N", scale=alt.Scale(domain=list(SERIES), range=list(SERIES.values())),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["Experiment", "Metric", alt.Tooltip("Score:Q", format=".3f")],
    ).properties(height=380, title="Accuracy and macro-F1 on 3,456 test articles")
    st.altair_chart(chart.configure_axis(gridColor="#eef0f2", domainColor="#c9ccd1", labelColor="#52514e",
                                         titleColor="#52514e").configure_view(stroke=None), width="stretch")

    st.markdown("""<div class="cards">
     <div class="card"><div class="t">🔤 Character n-grams win</div><div class="d">Kinyarwanda adds prefixes and suffixes to
     stems. Character pieces match all those forms; whole words don't.</div></div>
     <div class="card"><div class="t">⚖️ Class weights matter</div><div class="d">Without them, topics with ~30 articles
     (fashion, environment) were never predicted.</div></div>
     <div class="card"><div class="t">🧠 Word2Vec helps the BiLSTM</div><div class="d">+0.08 macro-F1 over random embeddings,
     and best on the rarest topics, but still below E3.</div></div>
     <div class="card"><div class="t">🤖 Bigger isn't better here</div><div class="d">The CPU-trained Transformer read only
     ~50 words and was undertrained. The simple model that reads the whole article wins.</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="h" style="margin-top:22px">F1 per topic and model</div>', unsafe_allow_html=True)
    heat = per_class.drop(columns="train docs").rename(columns=names).reset_index(names="Topic").melt(
        "Topic", var_name="Model", value_name="F1")
    topics = list(per_class.index)
    grid = alt.Chart(heat).mark_rect(cornerRadius=3, stroke="#ffffff", strokeWidth=2).encode(
        x=alt.X("Topic:N", sort=topics, title="topics, most to least training data",
                axis=alt.Axis(labelAngle=-40)),
        y=alt.Y("Model:N", sort=[names[m] for m in list(names)[1:]], title=None, axis=alt.Axis(labelLimit=220)),
        color=alt.Color("F1:Q", scale=alt.Scale(domain=[0, 1], range=["#cde2fb", "#2a78d6", "#0d366b"]),
                        legend=alt.Legend(title="F1", format=".1f")),
        tooltip=["Model", "Topic", alt.Tooltip("F1:Q", format=".2f")],
    )
    labels = grid.mark_text(fontSize=10).encode(
        text=alt.Text("F1:Q", format=".2f"), color=alt.condition("datum.F1 > 0.55", alt.value("#ffffff"), alt.value("#0b0b0b")))
    st.altair_chart((grid + labels).properties(height=280).configure_view(stroke=None)
                    .configure_axis(domainColor="#c9ccd1", labelColor="#52514e", titleColor="#52514e"), width="stretch")
    st.caption("Rare topics are hardest. The BiLSTM (E5) does best on the four smallest ones.")

with tab_data:
    raw, d = data["train_raw"], data["sizes"]
    st.markdown(f"""<div class="tiles">
      <div class="tile"><b>{raw["rows"]:,}</b><span>training rows in the release</span></div>
      <div class="tile"><b>{raw["duplicate_rows"]:,}</b><span>exact duplicates removed</span></div>
      <div class="tile"><b>{raw["conflicting_label_rows"]:,}</b><span>rows with two different labels removed</span></div>
      <div class="tile"><b>{d["train"]:,} / {d["val"]} / {d["test"]:,}</b><span>train / validation / test</span></div>
    </div>""", unsafe_allow_html=True)

    seen = by_source[by_source["seen in train"]]
    unseen = by_source[~by_source["seen in train"]]
    st.markdown(f"""<div class="cards">
     <div class="card"><div class="t">🔁 Half the training set was copies</div><div class="d">One article appeared 80 times.
     Keeping duplicates inflates scores: a random split of the raw data gives ~89% accuracy, the honest test ~76%.</div></div>
     <div class="card"><div class="t">🌍 New websites are harder</div><div class="d">The final model gets
     {seen["E3_char_tfidf_svm_balanced"].iloc[0]:.0%} on a website seen in training, but
     {unseen["E3_char_tfidf_svm_balanced"].min():.0%}–{unseen["E3_char_tfidf_svm_balanced"].max():.0%}
     on the two it never saw.</div></div>
    </div>""", unsafe_allow_html=True)

    counts = pd.DataFrame({"Topic": LABELS, "Articles": [data["class_counts"]["train"][l] for l in LABELS]})
    bars = alt.Chart(counts).mark_bar(cornerRadiusEnd=4, color="#2a78d6", height=16).encode(
        y=alt.Y("Topic:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False)),
        x=alt.X("Articles:Q", title="training articles", axis=alt.Axis(tickCount=5)),
        tooltip=["Topic", "Articles"],
    )
    text_lbl = bars.mark_text(align="left", dx=4, color="#52514e", fontSize=11).encode(text="Articles:Q")
    st.altair_chart((bars + text_lbl).properties(height=470, title="Training articles per topic")
                    .configure_axis(gridColor="#eef0f2", domainColor="#c9ccd1", labelColor="#52514e",
                                    titleColor="#52514e").configure_view(stroke=None), width="stretch")
    st.caption("Source: KINNEWS (Niyongabo et al., 2020), real Rwandan news, MIT licence.")

st.markdown(f'<div class="foot">Data: KINNEWS (Niyongabo et al., 2020) · <a href="{GITHUB}">Code &amp; notebook</a> · '
            f'<a href="https://huggingface.co/{MODEL_REPO}">Models</a></div>', unsafe_allow_html=True)
