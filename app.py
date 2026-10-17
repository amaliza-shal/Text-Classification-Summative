"""Gradio web app for the Kinyarwanda news classifier. Run: python app.py"""
import html

import gradio as gr
import pandas as pd

from src.config import FIGURES_DIR, KIN_LABELS, RESULTS_DIR
from src.inference import SVMModel, load_models

MODELS = load_models()
DEFAULT_MODEL = next(iter(MODELS))

# shortened excerpts from the KINNEWS test split, with their true labels
EXAMPLES = [
    ["Perezida Kagame yakiriye Amavubi muri Village Urugwiro mbere y’uko bakina na Congo. Perezida w’u Rwanda Paul Kagame, yakiriye ikipe y’igihugu, Amavubi, ayaha impamba y’impanuro izayafasha kwitwara neza mu mikino ya CHAN.", "sport"],
    ["Ubukungu bw’isi bwitezweho kuzamuka ku kigero cya 2.5 % muri 2020. Muri iyi raporo izwi nka Global Economic Prospects yashyizwe hanze kuri uyu wa Gatatu, Banki y’Isi ivuga ko izamuka ry’ubukungu mu bihugu byateye imbere rizaganyukaho 1.4 %.", "economy"],
    ["Abamotari bashobora gusabwa kugendana Alcohol bahanaguza muri Casque mu gukumira Coronavirus mu Rwanda.", "health"],
    ["U Rwanda ku mwanya wa mbere mu bihugu bizasurwa cyane muri Afurika mu 2020. Uru rutonde rugaragaza ibihugu 30 bifite ibintu bikurura abakerarugendo.", "tourism"],
    ["Leta yatangiye umushinga wo guhindura amazi ya Nyabarongo urubogobogo. Hatangiye guterwa ubwatsi, amashyamba, imigano.", "environment"],
    ["Umunyempano Sam Rushimisha yasohoye indirimbo nshya yise ‘Ubutunzi’. Rushimisha Sam ni umukirisitu ukora ivugabutumwa abicishije mu ndirimbo zo kuramya no guhimbaza Imana.", "religion"],
]

EXAMPLE_LABELS = {t: label for t, label in EXAMPLES}

THEME = gr.themes.Soft(
    primary_hue=gr.themes.Color(c50="#e6f6fc", c100="#c0e8f7", c200="#8dd6f1", c300="#55c2ea", c400="#26b1e4",
                                c500="#00a1de", c600="#0089bd", c700="#006f99", c800="#005675", c900="#003d53",
                                c950="#002838"),
    secondary_hue="green",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)

CSS = """
.gradio-container {max-width: 1150px !important; margin: auto;}
#banner {background: linear-gradient(100deg, #00a1de 0%, #0089bd 60%, #20603d 100%); color: white;
         border-radius: 16px; padding: 26px 30px; margin-bottom: 6px;}
#banner h1 {margin: 0 0 6px 0; font-size: 1.9rem; color: white;}
#banner p {margin: 0; opacity: .92; font-size: 1rem; color: white;}
#banner .flag {height: 6px; border-radius: 3px; margin-top: 16px; width: 140px; box-shadow: 0 0 0 2px rgba(255,255,255,.55);
               background: linear-gradient(90deg, #00a1de 0 50%, #fad201 50% 75%, #20603d 75% 100%);}
.result-card {border-radius: 14px; padding: 20px 22px; border: 1px solid var(--border-color-primary);
              background: var(--background-fill-secondary);}
.result-card .kicker {text-transform: uppercase; letter-spacing: .08em; font-size: .75rem; opacity: .7;}
.result-card .topic {font-size: 2rem; font-weight: 700; margin: 2px 0 0 0; text-transform: capitalize;}
.result-card .kin {font-size: 1.05rem; opacity: .8; margin-bottom: 12px;}
.result-card .bar {height: 10px; border-radius: 5px; background: var(--border-color-primary); overflow: hidden;}
.result-card .bar > div {height: 100%; background: linear-gradient(90deg, #00a1de, #20603d);}
.result-card .meta {font-size: .85rem; margin-top: 8px; opacity: .8;}
.pill {display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: .8rem; font-weight: 600; margin-top: 10px;}
.pill.ok {background: #dcfce7; color: #166534;}
.pill.bad {background: #fee2e2; color: #991b1b;}
.pill.warn {background: #fef9c3; color: #854d0e;}
.placeholder {opacity: .6; text-align: center; padding: 40px 10px;}
footer {display: none !important;}
"""

COMPARE_COLS = ["Model", "Prediction", "Top score"]
EMPTY_CARD = '<div class="result-card placeholder">Paste a Kinyarwanda news text and press <b>Classify</b>.</div>'


def result_card(scores, model_name, true_label, n_words):
    top = max(scores, key=scores.get)
    p = float(scores[top])
    is_svm = model_name == SVMModel.name
    score_name = "relative score" if is_svm else "confidence"
    parts = [
        '<div class="result-card">',
        '<div class="kicker">Predicted topic</div>',
        f'<div class="topic">{top}</div>',
        f'<div class="kin">{KIN_LABELS[top]}</div>',
        f'<div class="bar"><div style="width:{p * 100:.0f}%"></div></div>',
        f'<div class="meta">{score_name}: {p:.0%} &middot; {html.escape(model_name)}</div>',
    ]
    if n_words < 5:
        parts.append('<span class="pill warn">Very short input: longer text gives more reliable predictions</span> ')
    if not is_svm and p < 0.5:
        parts.append('<span class="pill warn">Low confidence: the text may fit several topics</span> ')
    if true_label:
        ok = true_label == top
        parts.append(f'<span class="pill {"ok" if ok else "bad"}">'
                     f'{"Correct" if ok else "Wrong"}: dataset label is {true_label}</span>')
    parts.append("</div>")
    return "".join(parts)


def classify(text, model_name):
    text = (text or "").strip()
    true_label = EXAMPLE_LABELS.get(text)  # only shown for the test-set examples
    if not text:
        return EMPTY_CARD, None, pd.DataFrame(columns=COMPARE_COLS)
    scores = MODELS[model_name](text)
    label_scores = {f"{l} ({KIN_LABELS[l]})": float(s) for l, s in scores.items()}

    rows = []
    for name, model in MODELS.items():
        s = model(text)
        top = max(s, key=s.get)
        rows.append([name, f"{top} ({KIN_LABELS[top]})", f"{float(s[top]):.0%}"])
    compare = pd.DataFrame(rows, columns=COMPARE_COLS)

    return result_card(scores, model_name, true_label, len(text.split())), label_scores, compare


def results_table():
    df = pd.read_csv(RESULTS_DIR / "experiments.csv")
    df = df[df["split"] == "test"].drop(columns="split").sort_values("macro_f1", ascending=False).round(3)
    return df.rename(columns={"model": "Experiment", "accuracy": "Accuracy", "macro_precision": "Macro P",
                              "macro_recall": "Macro R", "macro_f1": "Macro F1", "weighted_f1": "Weighted F1"})


with gr.Blocks(title="Kinyarwanda News Classifier") as demo:
    gr.HTML("""<div id="banner">
        <h1>Kinyarwanda News Topic Classifier</h1>
        <p>Assigns a Kinyarwanda news article to one of 14 topics, from politics and sport to tourism and religion.
        Trained on the KINNEWS corpus (Niyongabo et al., 2020).</p><div class="flag"></div></div>""")

    with gr.Tabs():
        with gr.Tab("Classify"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=6):
                    text = gr.Textbox(lines=9, max_lines=20, label="Kinyarwanda news text",
                                      placeholder="Paste a headline and/or article body in Kinyarwanda...")
                    model_name = gr.Radio(list(MODELS), value=DEFAULT_MODEL, label="Model")
                    true_label = gr.Textbox(visible=False, label="True label")
                    with gr.Row():
                        clear_btn = gr.Button("Clear", variant="secondary")
                        run_btn = gr.Button("Classify", variant="primary")
                with gr.Column(scale=5):
                    card = gr.HTML(EMPTY_CARD)
                    probs = gr.Label(num_top_classes=5, label="Top 5 topics", show_label=True)
            compare = gr.Dataframe(pd.DataFrame(columns=COMPARE_COLS), label="All models on this text",
                                   interactive=False, wrap=True)
            gr.Examples(EXAMPLES, inputs=[text, true_label], label="Examples from the test set (click one, then Classify)",
                        examples_per_page=6)

        with gr.Tab("Results"):
            gr.Markdown("### Test-set results (3,456 articles)\nMacro-F1 is the main metric because the 14 classes are very imbalanced.")
            gr.Dataframe(results_table(), interactive=False)
            with gr.Row():
                gr.Image(str(FIGURES_DIR / "per_class_f1.png"), label="Per-class F1", show_label=True)
            with gr.Row():
                gr.Image(str(FIGURES_DIR / "class_distribution.png"), label="Class distribution", show_label=True)

        with gr.Tab("About"):
            gr.Markdown("""
### How it works
1. The text is cleaned the same way as the training data (Unicode and apostrophe fixes; for the classic models also lowercasing and removal of numbers and punctuation).
2. The selected model scores the text against all 14 topics.
3. The topic with the highest score is shown, together with the top 5 and the predictions of the other models.

### Models
- **Char TF-IDF + LinearSVC (E3), default**: character 2-5-gram features with a linear SVM. Best test results (76.1% accuracy, 0.577 macro-F1). Its scores are relative, not probabilities.
- **AfroXLMR-mini (E6)**: a multilingual Transformer adapted to 17 African languages including Kinyarwanda (Alabi et al., 2022), fine-tuned on KINNEWS (67.0% accuracy). It reads only the first ~50 words and rarely predicts the small topics.
- **BiLSTM + Word2Vec (E5)**: bidirectional LSTM with attention pooling over Word2Vec embeddings trained on KINNEWS (67.4% accuracy, best on the rarest topics).

### Limitations
- Trained on news from 2010s-2020 Rwandan websites; other genres (social media, SMS) are out of domain.
- Rare topics (tourism, fashion, environment, culture, history, education) had only 30-200 training articles and are predicted less reliably.
- Some topics overlap (culture/history, fashion/entertainment), and the dataset labels follow each website's own categories.
- The app does not check that the input is Kinyarwanda.

**Data:** KINNEWS (MIT licence), deduplicated. **Code:** [GitHub](https://github.com/amaliza-shal/Text-Classification-Summative)
""")

    run_btn.click(classify, [text, model_name], [card, probs, compare])
    text.submit(classify, [text, model_name], [card, probs, compare])
    model_name.change(classify, [text, model_name], [card, probs, compare])
    clear_btn.click(lambda: ("", "", EMPTY_CARD, None, pd.DataFrame(columns=COMPARE_COLS)), None,
                    [text, true_label, card, probs, compare])

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS)
