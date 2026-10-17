# Kinyarwanda News Topic Classification (KINNEWS)

Assigning Kinyarwanda news articles to one of 14 topics (politics, sport, economy, health, entertainment, history, technology, tourism, culture, fashion, religion, environment, education, relationship). The models range from a TF-IDF baseline to a BiLSTM with Word2Vec embeddings and fine-tuned multilingual Transformers (AfroXLMR-mini and the mMiniLM model it was adapted from).

**Start with [`kinnews_classification.ipynb`](kinnews_classification.ipynb)**, which runs the whole project in one place: data audit, preprocessing, every experiment, evaluation, error analysis and inference.

| | |
|---|---|
| Live demo | _add Hugging Face Space link_ |
| Demo video | _add link_ |

## Problem

Kinyarwanda is one of Rwanda's official languages, but there are few labelled NLP datasets for it. Rwandan news sites publish a lot of Kinyarwanda text every day. A topic classifier can be used to route, tag and search that content, and topic labels are useful for building other Kinyarwanda NLP resources. The target users are media houses, archivists and researchers who work with Kinyarwanda text.

## Dataset

**KINNEWS** (Niyongabo et al., 2020, MIT licence): real Kinyarwanda news articles from 15 Rwandan news websites and 5 newspapers. The publishers' 48 original categories were grouped into 14 topics, and two native-speaker annotators labelled each article from its title and content. We use the `kinnews_raw` version (title + body, original casing) from the Hugging Face Hub (`andreniyongabo/kinnews_kirnews`).

While preparing the data we found several quality problems in the official release:

| | train | test |
|---|---|---|
| rows in the release | 17,014 | 4,254 |
| rows with fewer than 10 words | 0 | 4 |
| rows whose text appears under **two different labels** | 1,453 | 0 |
| exact duplicate rows | 8,098 | 791 |
| test articles also in train | - | 3 |
| **unique articles kept** | **7,463** | **3,456** |

More than half of the official training set is duplicates (one article appears 80 times). We removed duplicates, dropped articles with conflicting labels, and removed the test articles that also appear in train. We then took a stratified 10% of train as a validation set, which gives **6,716 train / 747 val / 3,456 test**. We kept the official test split rather than re-splitting, but because of the deduplication our numbers are not directly comparable with results reported on the raw release.

Other things to know about the data:
- **Strong class imbalance and train/test shift.** Politics has 1,590 training articles, while tourism, fashion and environment have 30-34 each. The test set has 2-3 times more of these rare classes than train (e.g. 90 fashion test articles vs 34 train). See `results/figures/class_distribution.png`.
- **The test set comes from different websites.** Train has articles from 18 sites. The test set comes from three: muhabura.rw (46%), igihe.com (35%) and kigalipost.com (19%), and the last two never appear in train. So most of the test set is a cross-source evaluation.
- **Mojibake** in about 70 articles (`â€™` instead of `’`), fixed in `fix_mojibake`.
- **Spelling variation.** Kinyarwanda writes elided vowels with an apostrophe (`n'abana`, `y'u Rwanda`), and r/l spelling varies (`abanyeshuri`/`abanyeshuli`, `gatolika`/`gatulika`).
- **Large, sparse vocabulary.** The language is agglutinative, so the vocabulary is large: 139,883 word types in train, 55% of them seen only once.

### Preprocessing (`src/preprocess.py`)
- `text` (for TF-IDF, Word2Vec, BiLSTM): fix mojibake, NFKC normalisation, lowercase, remove URLs and numbers, and turn apostrophes and punctuation into spaces, so that `n'abana` gives the token `abana`.
- `text_tf` (for Transformers): only mojibake/Unicode fixes and URL removal. Case and punctuation are kept because the subword tokenizer was trained with them.

## Methods

| ID | Model | Why |
|---|---|---|
| E0 | Majority class | Lower bound |
| E1 | Word 1-2 gram TF-IDF + logistic regression | Standard strong baseline for topic classification |
| E2 | E1 + balanced class weights | Does reweighting help the rare classes? |
| E3 | Char 2-5 gram TF-IDF + LinearSVC (balanced) | Sub-word features for an agglutinative language with spelling variation |
| E4 | BiLSTM + attention, random embeddings | Sequence model trained from scratch |
| E5 | BiLSTM + attention, Word2Vec init | Do embeddings trained on the corpus help with little data? |
| E6 | AfroXLMR-mini fine-tuned | Pretrained on 17 African languages, including Kinyarwanda |
| E7 | mMiniLM fine-tuned | The model AfroXLMR-mini started from: same architecture, no African-language adaptation |
| E8 | E6 with learning rate 1.5e-4 | E6 was still improving when it stopped: was it undertrained? |

**BiLSTM (`src/bilstm.py`).** Word-level vocabulary (min count 2) → 100-d embeddings → 1-layer BiLSTM (128 units per direction) → additive attention pooling over time steps → dropout 0.4 → linear layer over 14 classes. Input is truncated to 400 tokens, which covers the median article. Training uses Adam (lr 2e-3), batch size 32, up to 12 epochs, gradient clipping at 1.0 and class-weighted cross-entropy. The checkpoint with the best validation macro-F1 is kept.

**Word2Vec (`src/word2vec.py`).** Skip-gram, 100-d, window 5, 10 negative samples, 10 epochs, trained on the training split only.

**Transformers (`src/finetune.py`).** 12-layer, 384-d encoders with a classification head on the first-token (`<s>`) representation. Settings: max 128 subword tokens (the title comes first, so it is never truncated), lr 5e-5 (E8: 1.5e-4), batch size 16, 4 epochs, 10% linear warmup, weight decay 0.01, class-weighted loss. The best epoch is chosen by validation macro-F1. The word-embedding matrix (96M of 118M parameters) is frozen. Both choices were needed to train on a laptop CPU with 8 GB RAM (about 2-3 hours per model); with 256 tokens and trainable embeddings, one epoch took more than 2 hours and ran out of memory. The cost is that the Transformers only see the title and roughly the first 50 words of each article.

**Metrics.** Accuracy, macro precision/recall/F1 and weighted F1. **Macro-F1 is the main metric** because the classes are very imbalanced. Accuracy is dominated by the five large classes, while macro-F1 gives every topic the same weight.

## Results (test set, 3,456 articles)

| ID | Model | Accuracy | Macro P | Macro R | Macro F1 | Weighted F1 | Val macro F1 |
|---|---|---|---|---|---|---|---|
| E0 | Majority class | 0.091 | 0.006 | 0.071 | 0.012 | 0.015 | 0.027 |
| E1 | Word TF-IDF + LogReg | 0.691 | 0.567 | 0.460 | 0.411 | 0.657 | 0.579 |
| E2 | E1 + class weights | 0.719 | 0.679 | 0.526 | 0.487 | 0.699 | 0.646 |
| E3 | Char TF-IDF + LinearSVC | **0.761** | 0.654 | **0.610** | **0.577** | **0.753** | **0.736** |
| E4 | BiLSTM, random emb. | 0.629 | 0.560 | 0.488 | 0.467 | 0.648 | 0.616 |
| E5 | BiLSTM, Word2Vec emb. | 0.674 | 0.591 | 0.592 | 0.545 | 0.690 | 0.665 |
| E6 | AfroXLMR-mini, lr 5e-5 | 0.670 | 0.405 | 0.449 | 0.390 | 0.634 | 0.445 |
| E7 | mMiniLM, lr 5e-5 | stopped after 2.6 epochs: did not learn (val. loss 2.65 ≈ ln 14) | | | | | 0.027 |
| E8 | AfroXLMR-mini, lr 1.5e-4 | stopped after 1 epoch: collapsed to one class (val. loss 2.76) | | | | | 0.008 |

**Final model: E3** (character TF-IDF + LinearSVC). It is the default in the web app; E5 and E6 can be selected for comparison.

What the experiments show:
- **Class weighting matters most for the rare classes.** E1 → E2 raises macro-F1 from 0.41 to 0.49, mostly through the small classes. Without weighting, fashion and environment get F1 = 0.
- **Character n-grams beat word n-grams** (E2 → E3, macro-F1 0.49 → 0.58). Kinyarwanda attaches noun-class prefixes and verb affixes to the stem, so one word appears in many surface forms. Character n-grams share features across those forms, and also across spelling variants such as r/l.
- **Word2Vec initialisation helps the BiLSTM** (E4 → E5, macro-F1 0.47 → 0.55). The best validation epoch also comes earlier (7 vs 11). With only 6.7k training documents, embeddings learned from scratch get too few updates per word.
- **The Transformers are undertrained and sensitive to the learning rate.** E6 was still improving after 4 epochs (val. macro-F1 0.21 → 0.45) and reaches 67.0% test accuracy but only 0.390 macro-F1: it never predicts culture or fashion. mMiniLM without African-language training (E7) didn't learn at all at the same rate, and a 3x higher rate (E8) collapsed. With 128-token inputs (about 50 words) and frozen embeddings, which we needed to train on an 8 GB CPU, the TF-IDF model, which reads the whole article, wins.
- **The BiLSTM does not beat the linear model overall**, but it does better on the four rarest classes (culture, fashion, environment, tourism; `results/per_class_f1.csv`). Both BiLSTMs overfit: train loss reaches 0.02 while validation F1 levels off (`results/figures/bilstm_curves.png`).

### Why our numbers are lower than the KINNEWS paper

Niyongabo et al. (2020) report about 88.5% accuracy. A similar word TF-IDF + SVM reaches **89.2%** on a random 9:1 split of the *raw* training data, but in that split 57% of the held-out articles are duplicates of training articles. After deduplication the same model gets 77.8%, and on the official test set 70-73%. So scores near 88% need duplicate leakage between training and evaluation; our numbers use the stricter protocol. The check is in section 12.1 of the notebook.

## Error analysis

Section 13 of the notebook has per-class F1 for all models (`results/per_class_f1.csv`, `results/figures/per_class_f1.png`), the most frequent confusions with example titles, accuracy by website and by length, and the share of unseen words.

Main observations (E3 unless stated otherwise):
- **Source shift is the largest error factor.** Accuracy is 0.89 on muhabura.rw (a site seen in training), but 0.68 on igihe.com and 0.60 on kigalipost.com, which never appear in train. E5 scores almost the same as E3 on the unseen sites (0.67 / 0.59), but much lower on muhabura.rw (0.71). This suggests part of E3's lead comes from site-specific wording, not from the topic itself.
- **The test labels sometimes follow a different definition.** 13 of the 41 education → relationship errors are dream interpretations (*Ibisobanuro by'inzozi...*) or name meanings, which the test set labels education but which never appear under that label in train. Most of the 27 health → politics errors are crime and accident reports (murders, bodies found, arrests by RIB) that the test sites label health.
- **Overlapping categories.** Culture → history is the most frequent confusion (48 articles). Several of these are about the Rwandan monarchy or traditional ceremonies, some of them with titles starting with "Amateka" (history). Fashion → entertainment errors are mostly celebrity stories (Kardashian, Miss Rwanda contestants).
- **Long articles are harder.** Accuracy is about 0.79 up to 300 words, 0.68 for 500-1000 words and 0.53 above 1000 words. A likely reason is that long articles mix several topics. For the BiLSTM there is also truncation: it only reads the first 400 tokens.
- **OOV words are not the main cause.** The OOV rate is only slightly higher for misclassified test articles (5.1%) than for correct ones (4.5%).

## Design decisions and why

| Decision | Why |
|---|---|
| KINNEWS dataset | Real news text (not synthetic), labelled by native speakers, MIT licence, and the standard Kinyarwanda topic benchmark, so results can be compared. |
| Remove duplicates and conflicting labels | Over half the training rows were copies, which inflates scores and leaks into validation. When an article has two different labels, one of them must be wrong. |
| Keep the official test split | It's the benchmark split, and because half of it comes from unseen websites it tests what matters in practice: new sources. |
| Stratified 10% validation set | Choose epochs and checkpoints without touching the test set; stratified so that even the 30-article topics appear in it. |
| Two text versions (`text`, `text_tf`) | TF-IDF/Word2Vec need normalised word tokens; the Transformer tokenizer was trained on normal cased text with punctuation. |
| Macro-F1 as main metric | Classes are very unbalanced; macro-F1 counts every topic equally, so ignoring rare topics is punished. |
| E1 word TF-IDF + logistic regression as baseline | Simple, fast, well understood, strong for topic classification, and the same family as the KINNEWS paper's baselines. |
| Class weights (E2) | Without them the model never predicts fashion or environment (34 training articles each). |
| Character n-grams (E3) | Kinyarwanda attaches prefixes and suffixes to stems and has spelling variants; character n-grams share features across those forms. |
| Word2Vec trained on train only | No test text leaks in, and the tokens match ours exactly. |
| BiLSTM with attention (E4/E5) | Reads word order in both directions; attention learns which words signal the topic. |
| AfroXLMR-mini, with mMiniLM as control (E6/E7) | Pretrained on Kinyarwanda and small enough for a CPU; comparing it with the model it started from isolates the effect of the African-language training. |
| 128 tokens and frozen embeddings | Needed to fit training into 8 GB RAM on a CPU; the trade-off is documented as a limitation. |
| Stopping E7 and E8 early | Both stayed at chance level on validation; running them longer wouldn't change the conclusion and the deadline was close. |
| Final model = E3 | Best accuracy and macro-F1, fast, small, and easy to explain. |
| Gradio on Hugging Face Spaces | The assignment allows Gradio. The free Spaces CPU tier has enough memory for all three models plus PyTorch, the app gets a public link, and the models live next to the code. |

## Existing Kinyarwanda news classifiers

Several community models fine-tuned on KINNEWS are on the Hugging Face Hub (e.g. `henryparfait/kinnews-topic-classifier`, `Seziberagabriel/kinyarwanda-news-classifier`), and some of them also remove duplicates. What this project adds: the explanation of the gap to the paper's ~88% (duplicate leakage), the cross-source analysis (half the test set comes from unseen websites), the matched AfroXLMR-mini vs mMiniLM comparison, and an error analysis that traces many errors to inconsistent category definitions between websites.

## Repository layout

```
src/
  config.py          paths, label list, seed
  download.py        fetch KINNEWS parquet files
  preprocess.py      cleaning, deduplication, train/val/test split
  evaluate.py        metrics, confusion matrices, prediction dumps (used by every model)
  baselines.py       E0-E3
  word2vec.py        Word2Vec training
  bilstm.py          E4-E5
  finetune.py        E6-E8
  inference.py       loads saved models and scores new text (used by app and notebook)
kinnews_classification.ipynb     the full project, step by step, with outputs
app.py               Gradio web app
deploy_space.py      uploads the app to a Hugging Face Space
results/             metrics, per-model reports and predictions, training histories, figures
report/              the written report (HTML and PDF)
```

## Reproducing

```bash
python -m venv .venv && .venv/Scripts/activate   # Windows; use bin/activate on Linux/Mac
pip install -r requirements.txt

python -m src.download
python -m src.preprocess
python -m src.baselines            # E0-E3, ~4 min on CPU
python -m src.word2vec
python -m src.bilstm --emb random  # E4, ~30 min on CPU
python -m src.bilstm --emb w2v     # E5
python -m src.finetune --model Davlan/afro-xlmr-mini --name E6_afroxlmr_mini                               # ~1.5-2.5 h on CPU
python -m src.finetune --model nreimers/mMiniLMv2-L12-H384-distilled-from-XLMR-Large --name E7_mminilm   # E7
python -m src.finetune --model Davlan/afro-xlmr-mini --name E8_afroxlmr_mini_lr1.5e-4 --lr 1.5e-4          # E8
python app.py                      # http://127.0.0.1:7860
```

Then open `kinnews_classification.ipynb` for the analysis and figures, or open it directly in VS Code / Jupyter with the `.venv` kernel and run all cells. Cells that train the BiLSTMs and Transformers load the saved checkpoints if they exist; set `RETRAIN = True` to train from scratch.

To deploy: `hf auth login`, then `python deploy_space.py --space <username>/kinnews-classifier`.

All random seeds are fixed to 42.

## References

- Niyongabo, R. A., Qu, H., Kreutzer, J., & Huang, L. (2020). KINNEWS and KIRNEWS: Benchmarking cross-lingual text classification for Kinyarwanda and Kirundi. *COLING 2020*.
- Alabi, J. O., Adelani, D. I., Mosbach, M., & Klakow, D. (2022). Adapting pre-trained language models to African languages via multilingual adaptive fine-tuning. *COLING 2022*.
- Conneau, A., et al. (2020). Unsupervised cross-lingual representation learning at scale. *ACL 2020*.
- Wang, W., et al. (2021). MiniLMv2: Multi-head self-attention relation distillation for compressing pretrained Transformers. *Findings of ACL 2021*.
- Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013). Distributed representations of words and phrases and their compositionality. *NeurIPS 2013*.
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation, 9*(8).
- Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural machine translation by jointly learning to align and translate. *ICLR 2015*.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *JMLR, 12*.
- Wolf, T., et al. (2020). Transformers: State-of-the-art natural language processing. *EMNLP 2020: System Demonstrations*.
