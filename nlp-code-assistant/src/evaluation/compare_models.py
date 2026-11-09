"""Compare recorded test metrics; benchmark warm CPU inference on validation code."""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

from src.bug_detection.experiment_data import DEFAULT_SPLITS, load_splits
from src.bug_detection.split_dataset import ROOT
from src.inference.models import LoraPredictor, TfidfPredictor


def compare(split_dir, model_root, output, samples=30):
    if samples < 1:
        raise ValueError("samples must be positive")
    (_, validation, _), manifest = load_splits(split_dir)
    rows = []
    for name, folder, cls in [
        ("TF-IDF + LR", "tfidf_v2", TfidfPredictor),
        ("CodeBERT + LoRA", "codebert_lora", LoraPredictor),
    ]:
        directory = Path(model_root) / folder
        if not (directory / "metrics.json").exists():
            rows.append(
                {
                    "model": name,
                    "status": "untrained",
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "accuracy": None,
                    "latency_ms_median": None,
                }
            )
            continue
        report = json.loads((directory / "metrics.json").read_text())
        if report["split_manifest"] != manifest:
            raise ValueError(
                f"{name}: incompatible split manifest; results cannot be compared"
            )
        predictor = cls(directory)
        probe = validation.code.iloc[:samples].tolist()
        for code in probe[:3]:
            predictor.predict(code)
        timings = []
        for code in probe:
            started = time.perf_counter()
            predictor.predict(code)
            timings.append((time.perf_counter() - started) * 1000)
        metrics = report["test"]
        files = (
            list(directory.glob("*.joblib"))
            if folder == "tfidf_v2"
            else list((directory / "adapter").rglob("*"))
        )
        artifact_bytes = sum(path.stat().st_size for path in files if path.is_file())
        row = {
            "model": name,
            "status": "measured",
            "precision": metrics["bug_precision"],
            "recall": metrics["bug_recall"],
            "f1": metrics["bug_f1"],
            "accuracy": metrics["accuracy"],
            "latency_ms_median": statistics.median(timings),
            "latency_ms_mean": statistics.mean(timings),
            "samples": len(timings),
            "artifact_bytes": artifact_bytes,
        }
        if folder != "tfidf_v2":
            row["base_plus_adapter_parameter_bytes"] = sum(
                p.numel() * p.element_size() for p in predictor.model.parameters()
            )
            row["size_note"] = (
                "Adapter bytes exclude pretrained base; parameter bytes include the loaded base"
            )
        rows.append(row)
    result = {
        "models": rows,
        "protocol": "Warm CPU, batch=1, includes tokenization/vectorization, excludes loading; validation rows only",
        "machine": platform.platform(),
        "python": platform.python_version(),
        "split_manifest": manifest,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(rows, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split-dir", type=Path, default=DEFAULT_SPLITS)
    p.add_argument("--model-root", type=Path, default=ROOT / "models")
    p.add_argument(
        "--output", type=Path, default=ROOT / "results/model_comparison.json"
    )
    p.add_argument("--samples", type=int, default=30)
    a = p.parse_args()
    compare(a.split_dir, a.model_root, a.output, a.samples)


if __name__ == "__main__":
    main()
