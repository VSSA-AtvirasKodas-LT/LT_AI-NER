import os
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)

from sklearn.metrics import confusion_matrix, matthews_corrcoef
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import yaml
from copy import deepcopy
import argparse
from transformers.trainer_utils import set_seed

import json

from sklearn.metrics import (
    accuracy_score as sk_accuracy,
    precision_score as sk_precision,
    recall_score as sk_recall,
    f1_score as sk_f1,
)

from sklearn.model_selection import KFold
import seaborn as sns
import torch

torch.cuda.set_device(0)

def run_fold(
    fold_idx: int,
    train_list: list,
    test_list: list,
    model_checkpoint: str,
    id2label: dict,
    label2id: dict,
    tokenizer,
    base_output_dir: str,
    train_args_cfg: dict,
    seed: int,
):
    # tokenization
    train_ds = Dataset.from_list(train_list).map(lambda x: tokenize_and_align(x, tokenizer), batched=False)
    test_ds  = Dataset.from_list(test_list).map(lambda x: tokenize_and_align(x, tokenizer), batched=False)

    # model
    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        use_safetensors=True,
    )

    # collator
    collator = DataCollatorForTokenClassification(tokenizer)

    # per-fold output dir
    fold_out = os.path.join(base_output_dir, f"fold{fold_idx}")
    os.makedirs(fold_out, exist_ok=True)

    # TrainingArguments
    ta = TrainingArguments(
        output_dir=fold_out,
        eval_strategy=train_args_cfg.get("eval_strategy", "epoch"),
        save_strategy=train_args_cfg.get("save_strategy", "no"),  # usually "no" for CV speed
        learning_rate=float(train_args_cfg.get("learning_rate", 2e-5)),
        per_device_train_batch_size=int(train_args_cfg.get("per_device_train_batch_size", 8)),
        per_device_eval_batch_size=int(train_args_cfg.get("per_device_eval_batch_size", 8)),
        num_train_epochs=float(train_args_cfg.get("num_epochs", 10)),
        weight_decay=float(train_args_cfg.get("weight_decay", 0.01)),
        logging_dir=os.path.join(fold_out, "logs"),
        logging_steps=int(train_args_cfg.get("logging_steps", 10)),
        save_total_limit=int(train_args_cfg.get("save_total_limit", 2)),
        gradient_accumulation_steps=int(train_args_cfg.get("gradient_accumulation_steps", 1)),
        warmup_ratio=float(train_args_cfg.get("warmup_ratio", 0.0)),
        fp16=bool(train_args_cfg.get("fp16", False)),
        dataloader_num_workers=int(train_args_cfg.get("dataloader_num_workers", 0)),
        dataloader_pin_memory=bool(train_args_cfg.get("dataloader_pin_memory", False)),
        report_to=train_args_cfg.get("report_to", []),
        seed=seed,
        data_seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=ta,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=lambda x: compute_metrics(x, id2label, ta.output_dir),
    )

    trainer.train()

    # Evaluate on the held-out fold
    metrics = trainer.evaluate(eval_dataset=test_ds)  # returns dict with eval_*
    # Strip the "eval_" prefix for aggregation cleanliness
    cleaned = {k.removeprefix("eval_"): v for k, v in metrics.items()}
    # Save per-fold metrics
    with open(os.path.join(fold_out, "metrics.json"), "w") as f:
        json.dump(cleaned, f, indent=2)
    return cleaned
    
def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def merge_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    cfg = deepcopy(cfg)

    # Root-level overrides
    root_overrides = {
        "model_checkpoint": args.model_checkpoint,
        "output_dir": args.output_dir,
        "trainer_output_dir": args.trainer_output_dir,
        "train_dir": args.train_dir,
        "test_dir": args.test_dir,
        "seed": args.seed,
        "cross_validation": args.cross_validation,
        "cross_validation_folds": args.cross_validation_folds
    }
    for k, v in root_overrides.items():
        if v is not None:
            cfg[k] = v

    # Training block overrides
    t = cfg.setdefault("training", {})
    training_overrides = {
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
        "eval_strategy": args.eval_strategy,
        "save_strategy": args.save_strategy,
        "save_total_limit": args.save_total_limit,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_ratio": args.warmup_ratio,
        "fp16": args.fp16
    }
    for k, v in training_overrides.items():
        if v is not None:
            t[k] = v

    return cfg

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train NER with Hugging Face Transformers (config + CLI overrides).")
    p.add_argument("--config", type=str, default="config.json", help="Path to JSON config file.")

    # Root overrides
    p.add_argument("--model-checkpoint", type=str)
    p.add_argument("--output-dir", type=str)
    p.add_argument("--trainer-output-dir", type=str)
    p.add_argument("--train-dir", type=str)
    p.add_argument("--test-dir", type=str)
    p.add_argument("--seed", type=int)

    # Training overrides
    p.add_argument("--num-epochs", type=int)
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--per-device-train-batch-size", type=int)
    p.add_argument("--per-device-eval-batch-size", type=int)
    p.add_argument("--weight-decay", type=float)
    p.add_argument("--logging-steps", type=int)
    p.add_argument("--eval-strategy", type=str, choices=["no", "steps", "epoch"])
    p.add_argument("--save-strategy", type=str, choices=["no", "steps", "epoch"])
    p.add_argument("--save-total-limit", type=int)
    p.add_argument("--gradient-accumulation-steps", type=int)
    p.add_argument("--warmup-ratio", type=float)
    p.add_argument("--fp16", type=lambda s: s.lower() in {"1", "true", "yes", "y"}, help="Set true/false to override fp16.")
    p.add_argument("--cross-validation", dest="cross_validation", type=lambda s: s.lower() in {"1","true","yes","y"}, help="Enable K-fold cross validation")
    p.add_argument("--cross-validation-k", dest="cross_validation_folds", type=int, help="Number of folds for cross validation")

    return p


def read_conll(filepath):
    tokens, labels, examples = [], [], []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                if tokens:
                    examples.append({"tokens": tokens, "ner_tags": labels})
                    tokens, labels = [], []
            else:
                parts = line.split()
                if len(parts) != 2:
                    print(f"Skipping malformed line {line_num} in {filepath}: '{line}'")
                    continue
                token, tag = parts
                tokens.append(token)
                labels.append(tag)
        if tokens:
            examples.append({"tokens": tokens, "ner_tags": labels})
    return examples


def load_conll_folder(folder_path):
    all_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".conll"):
            full_path = os.path.join(folder_path, filename)
            examples = read_conll(full_path)
            all_data.extend(examples)
    return all_data


def load_data(train_dir, test_dir):
    train_data = load_conll_folder(train_dir)
    test_data = load_conll_folder(test_dir)

    # Extract label list
    unique_labels = sorted({label for ex in train_data for label in ex["ner_tags"]})
    label2id = {l: i for i, l in enumerate(unique_labels)}
    id2label = {i: l for l, i in label2id.items()}

    # Convert string labels to IDs
    for dataset in (train_data, test_data):
        for ex in dataset:
            ex["labels"] = [label2id[tag] for tag in ex["ner_tags"]]

    return {
        "dataset": dataset,
        "label2id": label2id,
        "id2label": id2label,
        "train_data": train_data,
        "test_data": test_data,
    }


def tokenize_and_align(example, tokenizer):
    tokenized = tokenizer(example["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    word_ids = tokenized.word_ids()

    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            labels.append(-100)
        elif word_idx != previous_word_idx:
            labels.append(example["labels"][word_idx])
        else:
            labels.append(-100)
        previous_word_idx = word_idx

    tokenized["labels"] = labels
    return tokenized


def align_predictions(predictions, label_ids, id2label):
    preds = np.argmax(predictions, axis=2)
    batch_size, seq_len = preds.shape
    out_preds, out_labels = [], []

    for i in range(batch_size):
        pred_labels = []
        true_labels = []
        for j in range(seq_len):
            if label_ids[i][j] != -100:
                pred_labels.append(id2label[preds[i][j]])
                true_labels.append(id2label[label_ids[i][j]])
        out_preds.append(pred_labels)
        out_labels.append(true_labels)

    return out_preds, out_labels


def extract_entities(label_seq):
    entities = []
    start, end, label = None, None, None
    for i, tag in enumerate(label_seq):
        if tag.startswith("B-"):
            if label is not None:
                entities.append((start, end, label))
            start = i
            end = i + 1
            label = tag[2:]
        elif tag.startswith("I-") and label == tag[2:]:
            end = i + 1
        else:
            if label is not None:
                entities.append((start, end, label))
                start, end, label = None, None, None
    if label is not None:
        entities.append((start, end, label))
    return entities


def compute_span_matches(pred_spans, true_spans, mode="exact"):
    tp, fp, fn = 0, 0, 0
    used = set()

    for ps in pred_spans:
        matched = False
        for i, ts in enumerate(true_spans):
            if ts[2] != ps[2]:  # labels must match
                continue

            if mode == "exact" and ps == ts:
                matched = True
                used.add(i)
                break

            elif mode == "overlap":
                if max(ps[0], ts[0]) < min(ps[1], ts[1]):
                    matched = True
                    used.add(i)
                    break

            elif mode == "union":
                if ps[0] <= ts[1] and ps[1] >= ts[0]:
                    matched = True
                    used.add(i)
                    break

        if matched:
            tp += 1
        else:
            fp += 1

    fn = len([ts for i, ts in enumerate(true_spans) if i not in used])
    return tp, fp, fn


def compute_metrics(p, id2label, output_dir):
    # p = (predictions, labels). align_predictions should
    #  - map to tag strings
    #  - drop/ignore -100 positions
    predictions, labels = p
    preds, refs = align_predictions(predictions, labels, id2label)  # lists of lists of tag strings

    # Flatten to 1D lists of tags
    # preds, refs = list[list[str]]
    flat_preds = [t for seq in preds for t in seq]
    flat_refs  = [t for seq in refs  for t in seq]

    token_accuracy  = sk_accuracy(flat_refs, flat_preds)
    token_precision = sk_precision(flat_refs, flat_preds, average="macro", zero_division=0)
    token_recall    = sk_recall(flat_refs, flat_preds, average="macro", zero_division=0)
    token_f1        = sk_f1(flat_refs, flat_preds, average="macro", zero_division=0)

    mcc = matthews_corrcoef(flat_refs, flat_preds)

    # --- Confusion matrix (avoid shape mismatch) ---
    all_labels = sorted(set(flat_refs) | set(flat_preds))  # UNION
    if all_labels:
        cm = confusion_matrix(flat_refs, flat_preds, labels=all_labels)
        cm_df = pd.DataFrame(cm, index=all_labels, columns=all_labels)
        # NOTE: Trainer expects only scalars in the returned dict,
        # so don't return cm_df here. You can log/save it elsewhere.
        cm_df.to_csv(f"{output_dir}/confusion_matrix.csv")

        plt.figure(figsize=(8, 6))
        # Draw heatmap without color bar
        ax = sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 10, "weight": "bold"}, 
                        linewidths=0.5, linecolor='gray')

        # Improve annotation contrast dynamically:
        for text in ax.texts:
            val = int(text.get_text())
            # Set text color: white if background is dark, otherwise black
            threshold = cm_df.values.max() / 2
            color = 'white' if val > threshold else 'black'
            text.set_color(color)

        plt.xlabel('Predicted Entity')
        plt.ylabel('True Entity')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/confusion_matrix.png")
        plt.close()
    else:
        cm_df = pd.DataFrame()

    # --- Span-level metrics ---
    def safe_div(a, b): return a / b if b else 0.0
    def scores(tp, fp, fn):
        precision = safe_div(tp, tp + fp)
        recall    = safe_div(tp, tp + fn)
        f1        = safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
        return precision, recall, f1

    exact_tp = exact_fp = exact_fn = 0
    overlap_tp = overlap_fp = overlap_fn = 0
    union_tp = union_fp = union_fn = 0

    for pred_seq, ref_seq in zip(preds, refs):
        pred_spans = extract_entities(pred_seq)  # -> [(start,end,type), ...]
        ref_spans  = extract_entities(ref_seq)

        tp, fp, fn = compute_span_matches(pred_spans, ref_spans, mode="exact")
        exact_tp += tp
        exact_fp += fp
        exact_fn += fn

        tp, fp, fn = compute_span_matches(pred_spans, ref_spans, mode="overlap")
        overlap_tp += tp
        overlap_fp += fp
        overlap_fn += fn

        tp, fp, fn = compute_span_matches(pred_spans, ref_spans, mode="union")
        union_tp += tp
        union_fp += fp
        union_fn += fn

    exact_p, exact_r, exact_f1 = scores(exact_tp, exact_fp, exact_fn)
    overlap_p, overlap_r, overlap_f1 = scores(overlap_tp, overlap_fp, overlap_fn)
    union_p, union_r, union_f1 = scores(union_tp, union_fp, union_fn)

    return {
        "token_accuracy": token_accuracy,
        "token_precision": token_precision,
        "token_recall": token_recall,
        "token_f1": token_f1,
        "entity_exact_precision": exact_p,
        "entity_exact_recall": exact_r,
        "entity_exact_f1": exact_f1,
        "entity_overlap_precision": overlap_p,
        "entity_overlap_recall": overlap_r,
        "entity_overlap_f1": overlap_f1,
        "entity_union_precision": union_p,
        "entity_union_recall": union_r,
        "entity_union_f1": union_f1,
        "matthews_coefficient": mcc,
    }


def eval_metrics(trainer, train_dataset, test_dataset, output_dir):
    # Final Evaluation
    final_predictions, _, metrics = trainer.predict(test_dataset)

    # Evaluate on test dataset
    test_preds, _, test_metrics = trainer.predict(test_dataset)

    # Evaluate on train dataset
    train_preds, _, train_metrics = trainer.predict(train_dataset)

    # Convert to DataFrames
    test_df = pd.DataFrame([test_metrics], index=["Test"])
    train_df = pd.DataFrame([train_metrics], index=["Train"])

    # Combine into a single summary table
    results_df = pd.concat([train_df, test_df])
    # print("\nEvaluation Metrics (Train vs Test):\n")
    # print(results_df.T.round(4))
    results_df.T.to_csv(f"{output_dir}/evaluation_metrics_results.csv", index=True)

    # Reformatting data for grouped bar plot
    metrics = results_df.T[:-3].index.tolist()
    train_values = results_df.T["Train"][:-3].tolist()
    test_values = results_df.T["Test"][:-3].tolist()

    x = range(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar([p - width / 2 for p in x], train_values, width, label="Train")
    ax.bar([p + width / 2 for p in x], test_values, width, label="Test")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Train vs Test Metrics (Grouped by Metric)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/evaluation_metrics_plot.png", dpi=600)


def eval_misclassified(trainer, tokenizer, test_dataset, id2label):
    # Step 1: Get predictions and aligned labels
    predictions, label_ids, _ = trainer.predict(test_dataset)
    preds, refs = align_predictions(predictions, label_ids, id2label)

    # Step 2: Decode tokens for each example
    print("Misclassified NER Samples:\n")
    max_display = 10
    shown = 0

    for i in range(len(test_dataset)):
        # Get input_ids for the current example
        input_ids = test_dataset[i]["input_ids"]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        # Get aligned predicted and true labels
        pred_labels = preds[i]
        true_labels = refs[i]

        # Filter mismatches (skip padding)
        mismatches = [
            (tok, pred, true)
            for tok, pred, true in zip(tokens, pred_labels, true_labels)
            if pred != true and true != "O" and tok not in tokenizer.all_special_tokens
        ]

        if mismatches:
            decoded_text = tokenizer.decode(input_ids, skip_special_tokens=True)
            print(f"Text: {decoded_text}")
            print("Mismatched Tokens:")
            for tok, pred, true in mismatches:
                decoded = tokenizer.convert_tokens_to_string([tok])
                print(f"  Token: {decoded:15} | Predicted: {pred:10} | True: {true}")
            print("-" * 60)
            shown += 1

        if shown >= max_display:
            break


def main():
    parser = build_arg_parser()
    cli_args = parser.parse_args()
    cfg = load_config(cli_args.config)
    cfg = merge_overrides(cfg, cli_args)

    model_checkpoint = cfg["model_checkpoint"]
    output_dir = cfg["output_dir"]
    trainer_output_dir = cfg["trainer_output_dir"]
    train_dir = cfg["train_dir"]
    test_dir = cfg["test_dir"]
    seed = cfg.get("seed", 42)
    cross_val = bool(cfg.get("cross_validation", False))
    k_folds = int(cfg.get("cross_validation_k", 5))

    set_seed(seed)

    data = load_data(train_dir, test_dir)

    label2id = data["label2id"]
    id2label = data["id2label"]

    # Put the token using !huggingface-cli login
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    train_dataset = Dataset.from_list(data["train_data"]).map(
        lambda x: tokenize_and_align(x, tokenizer),
        batched=False,
    )
    test_dataset = Dataset.from_list(data["test_data"]).map(
        lambda x: tokenize_and_align(x, tokenizer),
        batched=False,
    )

    if cross_val:
        # 1) Merge datasets
        all_items = list(data["train_data"]) + list(data["test_data"])
        n = len(all_items)
        indices = np.arange(n)

        # 2) KFold splitter (sequence-level)
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
        fold_metrics = []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(indices), start=1):
            train_list = [all_items[i] for i in train_idx]
            test_list  = [all_items[i] for i in test_idx]

            # Different seed per fold (optional)
            fold_seed = seed + fold_idx

            m = run_fold(
                fold_idx=fold_idx,
                train_list=train_list,
                test_list=test_list,
                model_checkpoint=model_checkpoint,
                id2label=id2label, label2id=label2id,
                tokenizer=tokenizer,
                base_output_dir=trainer_output_dir,    # folds saved under this dir
                train_args_cfg=cfg["training"],
                seed=fold_seed,
            )
            fold_metrics.append(m)

        # 3) Aggregate mean/std across folds (numeric keys only)

        # collect all keys that are numeric in all folds
        keys = [k for k in fold_metrics[0].keys()
                if all(isinstance(fm.get(k, None), (int, float)) for fm in fold_metrics)]

        means = {k: float(np.mean([fm[k] for fm in fold_metrics])) for k in keys}
        stds  = {k: float(np.std([fm[k] for fm in fold_metrics], ddof=1)) for k in keys}  # sample std

        summary = {"mean": means, "std": stds, "folds": fold_metrics}
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "cv_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        # Print a compact summary
        print("\n=== Cross-Validation Summary ===")
        for k in keys:
            print(f"{k}: {means[k]:.4f} ± {stds[k]:.4f}")

        # Skip the single train+eval path if CV was enabled
        return

    # Model Setup
    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        num_labels=len(data["label2id"]),
        id2label=data["id2label"],
        label2id=data["label2id"],
        use_safetensors=True,
    )

    # training args (with safe defaults to avoid freezing)
    t = cfg["training"]
    train_args = TrainingArguments(
        output_dir=trainer_output_dir,
        eval_strategy=t.get("eval_strategy", "epoch"),
        save_strategy=t.get("save_strategy", "epoch"),
        learning_rate=float(t.get("learning_rate", 2e-5)),
        per_device_train_batch_size=t.get("per_device_train_batch_size", 8),
        per_device_eval_batch_size=t.get("per_device_eval_batch_size", 8),
        num_train_epochs=t.get("num_epochs", 10),
        weight_decay=t.get("weight_decay", 0.01),
        logging_dir=f"{output_dir}/logs",
        logging_steps=t.get("logging_steps", 10),
        save_total_limit=t.get("save_total_limit", 2),
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 1),
        warmup_ratio=t.get("warmup_ratio", 0.0),
        fp16=bool(t.get("fp16", False)),
        dataloader_num_workers=t.get("dataloader_num_workers", 0),
        dataloader_pin_memory=bool(t.get("dataloader_pin_memory", False)),
        report_to=t.get("report_to", []),
    )

    # train the model
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=lambda x: compute_metrics(x, data["id2label"], train_args.output_dir),
    )

    trainer.train()

    eval_metrics(trainer, train_dataset, test_dataset, output_dir)
    eval_misclassified(trainer, tokenizer, test_dataset, data["id2label"])

if __name__ == "__main__":
    main()