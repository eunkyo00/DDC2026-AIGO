"""E0/E1/E3 reconstruction. Run on Colab GPU; saves the final epoch model."""
import argparse
import inspect
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, set_seed
from sklearn.metrics import f1_score
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core import ROOT, TARGETS, head_tail, fresh_dir, load_split, save_predictions, write_json, environment


class TextDataset(Dataset):
    def __init__(self, frame, tokenizer, method):
        self.labels = frame[TARGETS].to_numpy(dtype=np.float32)
        self.ids = []
        for text in frame.text.fillna("").astype(str):
            ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
            ids = head_tail(ids) if method == "head_tail" else ids[:510]
            self.ids.append([tokenizer.cls_token_id] + ids + [tokenizer.sep_token_id])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return {"input_ids": self.ids[index], "labels": self.labels[index]}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", choices=["e0", "e1", "e3"], required=True)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data/processed/calls")
    p.add_argument("--manifest", type=Path, default=ROOT / "data/splits/fixed/split_manifest.csv")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Select a Colab GPU runtime before training")
    out = fresh_dir(args.out_dir)
    set_seed(42)
    large = args.experiment == "e3"
    name = "klue/roberta-large" if large else "klue/roberta-base"
    method = "first_512" if args.experiment == "e0" else "head_tail"
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.model_max_length = 1_000_000
    train, valid = load_split(args.data_dir, args.manifest)
    train_ds, valid_ds = [TextDataset(frame, tokenizer, method) for frame in (train, valid)]
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=9,
        problem_type="multi_label_classification", id2label=dict(enumerate(TARGETS)),
        label2id={name: i for i, name in enumerate(TARGETS)})

    def collate(batch):
        padded = tokenizer.pad([{"input_ids": item["input_ids"]} for item in batch], return_tensors="pt")
        padded["labels"] = torch.tensor(np.stack([item["labels"] for item in batch]), dtype=torch.float32)
        return padded

    def metrics(prediction):
        logits, labels = prediction
        if isinstance(logits, tuple):
            logits = logits[0]
        return {"macro_f1": f1_score(labels, logits >= 0, average="macro", zero_division=0)}

    kw = dict(output_dir=str(out / "checkpoints"), num_train_epochs=4,
              per_device_train_batch_size=4 if large else 8,
              per_device_eval_batch_size=8 if large else 16,
              gradient_accumulation_steps=2 if large else 1,
              learning_rate=1e-5 if large else 2e-5, warmup_steps=1000,
              weight_decay=.01, logging_steps=200, fp16=True, report_to="none", seed=42,
              save_strategy="epoch", save_total_limit=1, load_best_model_at_end=False)
    key = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments).parameters else "evaluation_strategy"
    kw[key] = "epoch"
    trainer = Trainer(model=model, args=TrainingArguments(**kw), train_dataset=train_ds,
                      eval_dataset=valid_ds, data_collator=collate, compute_metrics=metrics)
    write_json(out / "run_config.json", {"experiment": args.experiment, "model": name,
        "input_method": method, "targets": TARGETS, "training_args": kw, "environment": environment()})
    trainer.train()
    trainer.save_model(str(out / "model"))
    tokenizer.model_max_length = 512
    tokenizer.save_pretrained(str(out / "model"))
    logits = trainer.predict(valid_ds).predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    save_predictions(out, valid, torch.sigmoid(torch.as_tensor(logits)).numpy())
    write_json(out / "training_log.json", trainer.state.log_history)
