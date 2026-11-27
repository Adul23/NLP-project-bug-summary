"""Compare TF-IDF + LR and CodeBERT + LoRA on the shared split.

Keeps the original output (recorded test metrics per model + warm CPU latency) and adds:
per-row validation/test predictions, the full metric set from bug_metrics (with bootstrap
CIs), validation-tuned thresholds, breakdowns by mutation type / code length / truncation,
McNemar's test and a paired bootstrap of the F1 difference, and efficiency numbers.

Discipline: thresholds are tuned on VALIDATION predictions only. Test predictions are only
scored, never used to choose anything.

CodeBERT inference over ~4,400 rows is slow on CPU. It runs on CUDA when available; on a
CPU-only machine it is skipped unless you pass --device cpu explicitly. --skip-predict
reuses existing prediction CSVs.
"""

import argparse
import json
from functools import partial
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.bug_detection.experiment_data import DEFAULT_SPLITS, load_splits
from src.bug_detection.split_dataset import ROOT
from src.evaluation import bug_metrics as bm
from src.inference.models import LoraPredictor, TfidfPredictor

MODELS = {
    "tfidf": {"name": "TF-IDF + LR", "folder": "tfidf_v2", "predictions": "tfidf_v2"},
    "codebert": {"name": "CodeBERT + LoRA", "folder": "codebert_lora", "predictions": "codebert"},
}
PREDICTION_COLUMNS = [
    "sample_id",
    "source_id",
    "label",
    "mutation_type",
    "predicted_label",
    "bug_probability",
    "n_tokens",
    "truncated",
]
COLAB_COMMAND = (
    "python -m src.evaluation.compare_models --models codebert --device cuda --batch-size 64 "
    "--output /content/colab_comparison.json"
)


# --- predictions ------------------------------------------------------------------------


def resolve_device(requested):
    try:
        import torch

        cuda = torch.cuda.is_available()
    except ImportError:
        cuda = False
    if requested == "cuda" and not cuda:
        raise ValueError("CUDA requested but unavailable")
    return "cuda" if requested == "auto" and cuda else ("cpu" if requested == "auto" else requested)


def token_counts(codes, tokenizer_dir):
    """CodeBERT token counts (with <s>/</s>) for length/truncation breakdowns, or None."""
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    except (ImportError, OSError, ValueError):
        return None
    return [len(ids) for ids in tokenizer(list(codes), truncation=False, verbose=False)["input_ids"]]


def tfidf_probabilities(predictor, codes, batch_size=None):
    features = predictor.vectorizer.transform(list(codes))
    return predictor.model.predict_proba(features)[:, 1], predictor.model.predict(features)


def codebert_probabilities(predictor, codes, batch_size=32, device="cpu", progress=True):
    """BUGGY probabilities in input order; batches are length-sorted to limit padding."""
    import torch

    model = predictor.model.to(device)
    codes = list(codes)
    lengths = [len(c) for c in codes]
    order = sorted(range(len(codes)), key=lambda i: lengths[i], reverse=True)
    probabilities = np.zeros(len(codes))
    started = time.perf_counter()
    for start in range(0, len(order), batch_size):
        rows = order[start : start + batch_size]
        inputs = predictor.tokenizer(
            [codes[i] for i in rows],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=predictor.config["max_length"],
            return_token_type_ids=False,
        ).to(device)
        with torch.inference_mode():
            probabilities[rows] = model(**inputs).logits.float().softmax(-1)[:, 1].cpu().numpy()
        if progress and (start // batch_size) % 10 == 0:
            done = min(start + batch_size, len(order))
            print(f"  codebert {done}/{len(order)} rows, {time.perf_counter() - started:.0f}s", flush=True)
    # LoraPredictor uses argmax, i.e. BUGGY when p(BUGGY) > p(CLEAN).
    return probabilities, (probabilities > 0.5).astype(int)


def prediction_frame(frame, probabilities, predicted, n_tokens, max_length):
    out = frame[["sample_id", "source_id", "label", "mutation_type"]].copy()
    out["predicted_label"] = np.asarray(predicted, dtype=int)
    out["bug_probability"] = np.asarray(probabilities, dtype=float)
    out["n_tokens"] = n_tokens if n_tokens is not None else pd.NA
    out["truncated"] = [n > max_length for n in n_tokens] if n_tokens is not None else pd.NA
    return out[PREDICTION_COLUMNS]


def read_predictions(path, frame):
    """Reuse a prediction CSV only if it covers exactly these rows with these labels."""
    saved = pd.read_csv(path, dtype={"source_id": str})
    merged = frame[["sample_id", "label"]].merge(saved, on="sample_id", how="left", suffixes=("", "_saved"))
    if merged.bug_probability.isna().any() or (merged.label != merged.label_saved).any():
        raise ValueError(f"{path} does not match the current split; delete it or drop --skip-predict")
    return saved.set_index("sample_id").loc[frame.sample_id].reset_index()


# --- efficiency -------------------------------------------------------------------------


def latency_profile(predict_one, probe, warmup=3):
    for code in probe[:warmup]:
        predict_one(code)
    timings = []
    for code in probe:
        started = time.perf_counter()
        predict_one(code)
        timings.append((time.perf_counter() - started) * 1000)
    return {
        "latency_ms_p50": statistics.median(timings),
        "latency_ms_p95": float(np.percentile(timings, 95)),
        "latency_ms_mean": statistics.mean(timings),
        "samples": len(timings),
    }


def throughput(predict_batch, codes, batch_size=32):
    predict_batch(codes[:batch_size])  # warm-up
    started = time.perf_counter()
    for start in range(0, len(codes), batch_size):
        predict_batch(codes[start : start + batch_size])
    seconds = time.perf_counter() - started
    return {"batch_size": batch_size, "rows": len(codes), "functions_per_second": len(codes) / seconds}


def parameter_counts(key, predictor):
    if key == "tfidf":
        coefficients = int(predictor.model.coef_.size + predictor.model.intercept_.size)
        return {
            "trainable_parameters": coefficients,
            "total_parameters": coefficients,
            "vocabulary_size": len(predictor.vectorizer.vocabulary_),
            "note": "logistic-regression weights; the TF-IDF vocabulary is fitted, not trained",
        }
    trainable = sum(
        p.numel()
        for n, p in predictor.model.named_parameters()
        if "lora_" in n or "modules_to_save" in n
    )
    total = sum(p.numel() for p in predictor.model.parameters())
    return {
        "trainable_parameters": int(trainable),
        "total_parameters": int(total),
        "trainable_share": trainable / total,
        "note": "trainable = LoRA matrices + the saved classifier head copy",
    }


def training_time(directory):
    config_path = Path(directory) / "training_config.json"
    if config_path.exists():
        runtime = json.loads(config_path.read_text()).get("runtime") or {}
        if runtime.get("train_seconds") is not None:
            return {"seconds": runtime["train_seconds"], "source": "training_config.json runtime"}
    return {"seconds": None, "source": "not recorded in the model folder"}


# --- evaluation -------------------------------------------------------------------------


def evaluate_model(validation_rows, test_rows, n_bootstrap, quartile_groups, quartile_labels, max_length):
    tuned = bm.best_f1_threshold(validation_rows.label, validation_rows.bug_probability)
    y_true = test_rows.label.to_numpy()
    y_prob = test_rows.bug_probability.to_numpy()
    y_pred = (y_prob >= 0.5).astype(int)
    result = {
        "test": bm.compute_bug_metrics(y_true, y_pred, y_prob, test_rows.mutation_type, n_bootstrap=n_bootstrap),
        "threshold": {
            "chosen_on": "validation",
            "validation_tuned_threshold": tuned["threshold"],
            "validation_bug_f1_at_tuned": tuned["validation_bug_f1"],
            "analysis_on_test": bm.threshold_analysis(
                y_true, y_prob, tuned["threshold"], test_rows.mutation_type, n_bootstrap=0
            ),
        },
        "validation_at_0.5": bm.compute_bug_metrics(
            validation_rows.label, (validation_rows.bug_probability >= 0.5).astype(int),
            validation_rows.bug_probability, n_bootstrap=0,
        ),
    }
    if quartile_groups is not None:
        result["by_code_length"] = bm.breakdown(y_true, y_pred, quartile_groups, y_prob, order=quartile_labels)
        truncated = np.where(test_rows.n_tokens.to_numpy() > max_length, f"truncated (> {max_length} tokens)", "fits")
        result["by_truncation"] = bm.breakdown(
            y_true, y_pred, truncated, y_prob, order=["fits", f"truncated (> {max_length} tokens)"]
        )
    return result


def compare(
    split_dir,
    model_root,
    output,
    samples=30,
    *,
    models=tuple(MODELS),
    device="auto",
    batch_size=32,
    skip_predict=False,
    n_bootstrap=bm.N_BOOTSTRAP,
    throughput_rows=64,
    report=None,
    allow_cpu_codebert=False,
):
    if samples < 1:
        raise ValueError("samples must be positive")
    output = Path(output)
    results_dir = output.parent
    results_dir.mkdir(parents=True, exist_ok=True)
    (_, validation, test), manifest = load_splits(split_dir)
    device = resolve_device(device)
    codebert_dir = Path(model_root) / MODELS["codebert"]["folder"]
    max_length = 512
    if (codebert_dir / "training_config.json").exists():
        max_length = int(json.loads((codebert_dir / "training_config.json").read_text()).get("max_length", 512))
    n_tokens = {
        name: token_counts(frame.code, codebert_dir / "adapter")
        for name, frame in (("validation", validation), ("test", test))
    }
    quartile_groups = quartile_labels = None
    if n_tokens["test"] is not None:
        quartile_groups, quartile_labels = bm.length_quartiles(n_tokens["test"])

    rows, evaluation, predictions, notes = [], {}, {}, []
    for key in models:
        spec = MODELS[key]
        directory = Path(model_root) / spec["folder"]
        if not (directory / "metrics.json").exists():
            rows.append(
                {
                    "model": spec["name"],
                    "status": "untrained",
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "accuracy": None,
                    "latency_ms_median": None,
                }
            )
            continue
        recorded = json.loads((directory / "metrics.json").read_text())
        if recorded["split_manifest"] != manifest:
            raise ValueError(f"{spec['name']}: incompatible split manifest; results cannot be compared")
        predictor = TfidfPredictor(directory) if key == "tfidf" else LoraPredictor(directory)

        # Original protocol: warm CPU, batch 1, 30 validation rows, includes tokenization.
        probe = validation.code.iloc[:samples].tolist()
        cpu_latency = latency_profile(predictor.predict, probe)
        if key == "tfidf":
            batch_fn = partial(tfidf_probabilities, predictor)
        else:
            batch_fn = partial(
                codebert_probabilities, predictor, batch_size=batch_size, device="cpu", progress=False
            )
        efficiency = {
            "cpu_batch1": cpu_latency,
            "cpu_throughput": throughput(batch_fn, validation.code.iloc[:throughput_rows].tolist(), 32),
            "cuda_batch1": None,
            "cuda_throughput": None,
            "parameters": parameter_counts(key, predictor),
            "training_time": training_time(directory),
        }
        if key == "codebert" and device == "cuda":
            predictor.model.to("cuda")
            efficiency["cuda_batch1"] = latency_profile(predictor.predict, probe)
            efficiency["cuda_throughput"] = throughput(
                partial(codebert_probabilities, predictor, batch_size=32, device="cuda", progress=False),
                validation.code.iloc[: max(throughput_rows, 512)].tolist(),
                32,
            )
            predictor.model.to("cpu")

        files = {
            split: results_dir / f"{spec['predictions']}_{split}_predictions.csv"
            for split in ("validation", "test")
        }
        split_rows = {}
        for split, frame in (("validation", validation), ("test", test)):
            path = files[split]
            if skip_predict and path.exists():
                split_rows[split] = read_predictions(path, frame)
                continue
            if key == "codebert" and device == "cpu" and not allow_cpu_codebert:
                notes.append(
                    "CodeBERT predictions skipped: no CUDA here. In Colab (repo root, with "
                    f"data/processed/stratified_v1 and models/codebert_lora) run `{COLAB_COMMAND}`, copy "
                    "results/codebert_validation_predictions.csv and results/codebert_test_predictions.csv "
                    "back into results/, then run `python -m src.evaluation.compare_models --skip-predict` "
                    "locally. On CPU instead: `--device cpu --batch-size 1` (batch 1 is faster than "
                    "batch 32 on this CPU; see the latency numbers)."
                )
                break
            print(f"predicting {spec['name']} on {split} ({len(frame)} rows, {device})", flush=True)
            if key == "tfidf":
                probabilities, predicted = tfidf_probabilities(predictor, frame.code)
            else:
                probabilities, predicted = codebert_probabilities(predictor, frame.code, batch_size, device)
            split_rows[split] = prediction_frame(frame, probabilities, predicted, n_tokens[split], max_length)
            split_rows[split].to_csv(path, index=False)
        if len(split_rows) == 2:
            predictions[key] = split_rows
            evaluation[key] = evaluate_model(
                split_rows["validation"], split_rows["test"], n_bootstrap, quartile_groups, quartile_labels, max_length
            )
            evaluation[key]["prediction_files"] = {s: str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p) for s, p in files.items()}

        files_on_disk = (
            list(directory.glob("*.joblib")) if key == "tfidf" else list((directory / "adapter").rglob("*"))
        )
        test_metrics = recorded["test"]
        row = {
            "model": spec["name"],
            "status": "measured",
            "precision": test_metrics["bug_precision"],
            "recall": test_metrics["bug_recall"],
            "f1": test_metrics["bug_f1"],
            "accuracy": test_metrics["accuracy"],
            "latency_ms_median": cpu_latency["latency_ms_p50"],
            "latency_ms_mean": cpu_latency["latency_ms_mean"],
            "latency_ms_p95": cpu_latency["latency_ms_p95"],
            "samples": cpu_latency["samples"],
            "artifact_bytes": sum(p.stat().st_size for p in files_on_disk if p.is_file()),
            "evaluated": key in evaluation,
            "efficiency": efficiency,
        }
        if key == "codebert":
            row["base_plus_adapter_parameter_bytes"] = sum(
                p.numel() * p.element_size() for p in predictor.model.parameters()
            )
            row["size_note"] = "Adapter bytes exclude pretrained base; parameter bytes include the loaded base"
        rows.append(row)

    paired = None
    if {"tfidf", "codebert"} <= set(predictions):
        a, b = predictions["tfidf"]["test"], predictions["codebert"]["test"]
        y_true = a.label.to_numpy()
        pred_a = (a.bug_probability.to_numpy() >= 0.5).astype(int)
        pred_b = (b.bug_probability.to_numpy() >= 0.5).astype(int)
        paired = {
            "model_a": MODELS["tfidf"]["name"],
            "model_b": MODELS["codebert"]["name"],
            "decision_threshold": 0.5,
            "mcnemar": bm.mcnemar(y_true, pred_a, pred_b),
            "paired_bootstrap_bug_f1": bm.paired_bootstrap_f1(y_true, pred_a, pred_b, n_bootstrap),
        }
    elif len(models) > 1:
        notes.append("McNemar and the paired bootstrap need test predictions from both models.")

    result = {
        "models": rows,
        "protocol": "Warm CPU, batch=1, includes tokenization/vectorization, excludes loading; validation rows only",
        "machine": platform.platform(),
        "python": platform.python_version(),
        "split_manifest": manifest,
        "evaluation": evaluation,
        "paired_comparison": paired,
        "evaluation_protocol": {
            "test_rows": len(test),
            "validation_rows": len(validation),
            "threshold_selection": "maximize BUGGY F1 on validation predictions; never on test",
            "headline_threshold": 0.5,
            "bootstrap": {"resamples": n_bootstrap, "seed": bm.SEED, "ci": "95% percentile"},
            "calibration_bins": 10,
            "max_length_tokens": max_length,
            "code_length_quartile_edges_from": "test rows, CodeBERT tokenizer counts",
            "throughput": f"batch 32 over the first {throughput_rows} validation rows",
            "device_for_predictions": device,
        },
        "notes": notes,
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (report or results_dir / "evaluation_report.md").write_text(render_report(result))
    print(json.dumps(rows, indent=2, default=str)[:2000])
    for note in notes:
        print("NOTE:", note)
    return result


# --- markdown report --------------------------------------------------------------------


def _f(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def _ci(metrics, key, digits=3):
    value = metrics.get(key)
    ci = ((metrics.get("bootstrap") or {}).get("ci") or {}).get(key)
    if value is None:
        return "—"
    return f"{value:.{digits}f} [{ci['low']:.{digits}f}, {ci['high']:.{digits}f}]" if ci else f"{value:.{digits}f}"


def render_report(result):
    lines = [
        "# Bug detection evaluation report",
        "",
        "Generated by `python -m src.evaluation.compare_models`. Every number below comes from",
        "`results/model_comparison.json` and the prediction CSVs it lists.",
        "",
        f"Test rows: {result['evaluation_protocol']['test_rows']}. Headline metrics use threshold 0.5. "
        "95% CIs: percentile bootstrap, 1000 resamples, seed 42. Thresholds are tuned on validation only.",
        "",
    ]
    evaluation = result["evaluation"]
    names = {row["model"]: row for row in result["models"]}
    keys = [k for k in MODELS if k in evaluation]
    if keys:
        lines += ["## Main metrics (test, threshold 0.5)", ""]
        header = ["Metric"] + [MODELS[k]["name"] for k in keys]
        lines += ["| " + " | ".join(header) + " |", "|---" + "|---:" * len(keys) + "|"]
        spec = [
            ("Accuracy", "accuracy", True),
            ("Balanced accuracy", "balanced_accuracy", False),
            ("BUGGY precision", "bug_precision", False),
            ("BUGGY recall", "bug_recall", False),
            ("BUGGY F1", "bug_f1", True),
            ("CLEAN F1", "clean_f1", False),
            ("Macro-F1", "macro_f1", False),
            ("Specificity (TNR)", "specificity", False),
            ("False positive rate", "false_positive_rate", False),
            ("False negative rate", "false_negative_rate", False),
            ("MCC", "mcc", True),
            ("Cohen's kappa", "cohen_kappa", False),
            ("ROC-AUC", "roc_auc", True),
            ("PR-AUC (baseline = positive rate)", "pr_auc", True),
        ]
        for label, key, with_ci in spec:
            cells = [
                _ci(evaluation[k]["test"], key) if with_ci else _f(evaluation[k]["test"].get(key)) for k in keys
            ]
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append(
            "| PR-AUC random baseline | " + " | ".join(_f(evaluation[k]["test"]["pr_auc_baseline"]) for k in keys) + " |"
        )
        for label, key in (("Brier score (lower is better)", "brier"), ("Log loss", "log_loss"), ("ECE, 10 bins", "ece")):
            lines.append(
                f"| {label} | " + " | ".join(_f((evaluation[k]["test"]["calibration"] or {}).get(key)) for k in keys) + " |"
            )
        for label, key in (("CPU latency p50, batch 1 (ms)", "latency_ms_p50"), ("CPU latency p95, batch 1 (ms)", "latency_ms_p95")):
            lines.append(
                f"| {label} | "
                + " | ".join(_f(names[MODELS[k]["name"]]["efficiency"]["cpu_batch1"][key], 2) for k in keys)
                + " |"
            )
        lines.append(
            "| CPU throughput, batch 32 (functions/s) | "
            + " | ".join(_f(names[MODELS[k]["name"]]["efficiency"]["cpu_throughput"]["functions_per_second"], 1) for k in keys)
            + " |"
        )
        lines += ["", "Values in brackets are 95% bootstrap confidence intervals.", ""]

        lines += ["## Recall per mutation type and clean false positives (test, threshold 0.5)", ""]
        lines += ["| Rows | " + " | ".join(MODELS[k]["name"] for k in keys) + " |", "|---" + "|---:" * len(keys) + "|"]
        types = [t for t in bm.MUTATION_TYPES if t in evaluation[keys[0]]["test"]["by_mutation_type"]]
        for t in types:
            cells = []
            for k in keys:
                m = evaluation[k]["test"]["by_mutation_type"][t]
                cells.append(f"{_f(m['recall'])} ({m['detected']}/{m['n']})")
            lines.append(f"| {t.replace('_', ' ')} recall | " + " | ".join(cells) + " |")
        cells = []
        for k in keys:
            m = evaluation[k]["test"]["by_mutation_type"]["clean"]
            cells.append(f"{_f(m['false_positive_rate'])} ({m['false_positives']}/{m['n']})")
        lines += ["| clean rows flagged BUGGY (FPR) | " + " | ".join(cells) + " |", ""]

        lines += ["## Threshold analysis (tuned on validation, applied to test)", ""]
        lines += ["| Model | Threshold | Test BUGGY P | Test BUGGY R | Test BUGGY F1 | Test MCC |", "|---|---:|---:|---:|---:|---:|"]
        for k in keys:
            for name, block in evaluation[k]["threshold"]["analysis_on_test"].items():
                m = block["metrics"]
                lines.append(
                    f"| {MODELS[k]['name']} ({name.replace('_', ' ')}) | {block['threshold']:.3f} | "
                    f"{_f(m['bug_precision'])} | {_f(m['bug_recall'])} | {_f(m['bug_f1'])} | {_f(m['mcc'])} |"
                )
        lines.append("")

        lines += ["## Confusion matrices (test, threshold 0.5)", ""]
        for k in keys:
            cm = evaluation[k]["test"]["confusion_matrix"]
            (tn, fp), (fn, tp) = cm["counts"]
            (tnr, fpr), (fnr, tpr) = cm["row_normalized"]
            lines += [
                f"**{MODELS[k]['name']}** (rows = actual, columns = predicted)",
                "",
                "| | Pred. CLEAN | Pred. BUGGY |",
                "|---|---:|---:|",
                f"| Actual CLEAN | {tn} ({tnr:.1%}) | {fp} ({fpr:.1%}) |",
                f"| Actual BUGGY | {fn} ({fnr:.1%}) | {tp} ({tpr:.1%}) |",
                "",
            ]

        if any("by_code_length" in evaluation[k] for k in keys):
            lines += ["## By code length (CodeBERT token-count quartiles of the test rows)", ""]
            lines += ["| Group | n | " + " | ".join(f"{MODELS[k]['name']} F1 / acc." for k in keys) + " |", "|---|---:" + "|---:" * len(keys) + "|"]
            for group in ("by_code_length", "by_truncation"):
                for name, first in evaluation[keys[0]].get(group, {}).items():
                    cells = [
                        f"{_f(evaluation[k][group][name]['bug_f1'])} / {_f(evaluation[k][group][name]['accuracy'])}"
                        if name in evaluation[k].get(group, {})
                        else "—"
                        for k in keys
                    ]
                    lines.append(f"| {name} | {first['n']} | " + " | ".join(cells) + " |")
            lines.append("")

    paired = result.get("paired_comparison")
    if paired:
        mc, boot = paired["mcnemar"], paired["paired_bootstrap_bug_f1"]
        lines += [
            "## Paired comparison on the same test rows (threshold 0.5)",
            "",
            f"- McNemar: TF-IDF right / CodeBERT wrong = {mc['b']}, TF-IDF wrong / CodeBERT right = {mc['c']}; "
            f"exact p = {mc['exact_p']:.3g}, continuity-corrected chi² = {mc['chi2']:.2f} (p = {mc['chi2_p']:.3g}).",
            f"- BUGGY F1 difference (CodeBERT − TF-IDF) = {boot['observed']:.3f}, "
            f"95% paired bootstrap CI [{boot['ci_low']:.3f}, {boot['ci_high']:.3f}].",
            "",
        ]

    lines += [
        "## Efficiency (warm CPU)",
        "",
        "| Model | Latency p50 (ms) | Latency p95 (ms) | Mean (ms) | Batch-32 throughput (fn/s) "
        "| Trainable params | Total params | Artifact size | Training time |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in result["models"]:
        if row.get("status") != "measured":
            continue
        eff = row["efficiency"]
        params = eff["parameters"]
        seconds = eff["training_time"]["seconds"]
        cpu = eff["cpu_batch1"]
        lines.append(
            f"| {row['model']} | {cpu['latency_ms_p50']:.2f} | {cpu['latency_ms_p95']:.2f} | "
            f"{cpu['latency_ms_mean']:.2f} | {eff['cpu_throughput']['functions_per_second']:.1f} | "
            f"{params['trainable_parameters']:,} | {params['total_parameters']:,} | "
            f"{row['artifact_bytes'] / 1e6:.1f} MB | "
            f"{'not recorded' if seconds is None else f'{seconds / 60:.1f} min'} |"
        )
    lines += [
        "",
        f"Latency: {result['protocol']}. Throughput: {result['evaluation_protocol']['throughput']}.",
        "",
    ]
    if result["notes"]:
        lines += ["## Notes", ""] + [f"- {note}" for note in result["notes"]] + [""]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split-dir", type=Path, default=DEFAULT_SPLITS)
    p.add_argument("--model-root", type=Path, default=ROOT / "models")
    p.add_argument("--output", type=Path, default=ROOT / "results/model_comparison.json")
    p.add_argument("--samples", type=int, default=30, help="rows for the batch-1 latency benchmark")
    p.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="CodeBERT prediction device; auto skips CodeBERT predictions when there is no CUDA",
    )
    p.add_argument("--batch-size", type=int, default=32, help="CodeBERT prediction batch size")
    p.add_argument("--skip-predict", action="store_true", help="reuse existing prediction CSVs")
    p.add_argument("--bootstrap", type=int, default=bm.N_BOOTSTRAP)
    a = p.parse_args()
    compare(
        a.split_dir,
        a.model_root,
        a.output,
        a.samples,
        models=tuple(a.models),
        device=a.device,
        batch_size=a.batch_size,
        skip_predict=a.skip_predict,
        n_bootstrap=a.bootstrap,
        allow_cpu_codebert=a.device == "cpu",
    )


if __name__ == "__main__":
    main()
