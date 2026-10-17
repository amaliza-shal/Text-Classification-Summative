"""Fine-tune a pretrained Transformer encoder on KINNEWS (~2-3 h per model on a laptop CPU).

python -m src.finetune --model Davlan/afro-xlmr-mini --name E6_afroxlmr_mini
python -m src.finetune --model nreimers/mMiniLMv2-L12-H384-distilled-from-XLMR-Large --name E7_mminilm

AfroXLMR-mini is mMiniLMv2 with extra MLM training on 17 African languages (Alabi et al., 2022),
so E6 vs E7 shows what that adaptation adds.
"""
import argparse

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainingArguments)

from src.config import DATA_PROCESSED, ID2LABEL, LABEL2ID, LABELS, MODELS_DIR, RESULTS_DIR, SEED
from src.evaluate import evaluate


class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy (same weighting as the BiLSTM)."""

    def __init__(self, class_weights, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = torch.nn.functional.cross_entropy(
            outputs.logits, labels, weight=self.class_weights.to(outputs.logits.device))
        return (loss, outputs) if return_outputs else loss


def train(model_name="Davlan/afro-xlmr-mini", name="E6_afroxlmr_mini", max_len=128, epochs=4, lr=5e-5,
          batch_size=16):
    train, val, test = (pd.read_csv(DATA_PROCESSED / f"{s}.csv") for s in ("train", "val", "test"))
    tok = AutoTokenizer.from_pretrained(model_name)

    def to_ds(df):
        ds = Dataset.from_dict({"text": df["text_tf"].tolist(), "labels": df["label"].map(LABEL2ID).tolist()})
        # title comes first in text_tf, so head truncation always keeps it
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=max_len), batched=True)

    ds_train, ds_val, ds_test = to_ds(train), to_ds(val), to_ds(test)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID, dtype=torch.float32)
    # The 250k x 384 word-embedding matrix is ~80% of the parameters. Freezing it keeps the
    # pretrained subword vectors, and makes CPU training fit in memory.
    for p in model.base_model.embeddings.word_embeddings.parameters():
        p.requires_grad = False

    counts = np.bincount(train["label"].map(LABEL2ID), minlength=len(LABELS))
    class_weights = torch.tensor(len(train) / (len(LABELS) * counts), dtype=torch.float)

    def metrics(p):
        return {"macro_f1": f1_score(p.label_ids, p.predictions.argmax(-1), average="macro")}

    out_dir = MODELS_DIR / name
    targs = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=64,
        warmup_steps=0.1,  # float = fraction of total steps
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        fp16=torch.cuda.is_available(),
        seed=SEED,
        report_to="none",
        logging_steps=50,
    )
    trainer = WeightedTrainer(class_weights=class_weights, model=model, args=targs,
                              train_dataset=ds_train, eval_dataset=ds_val,
                              data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics)
    trainer.train()
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    pd.DataFrame(trainer.state.log_history).to_csv(RESULTS_DIR / f"{name}_history.csv", index=False)

    for split, df, ds in (("val", val, ds_val), ("test", test, ds_test)):
        pred = trainer.predict(ds).predictions.argmax(-1)
        evaluate(name, df, [LABELS[i] for i in pred], split=split)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", dest="model_name", default="Davlan/afro-xlmr-mini")
    ap.add_argument("--name", default="E6_afroxlmr_mini")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch_size", type=int, default=16)
    train(**vars(ap.parse_args()))
