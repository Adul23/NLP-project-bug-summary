"""Benchmark the old vs the optimized CodeBERT LoRA training settings.

Runs a fixed number of optimizer steps per configuration and prints steps/sec, samples/sec
and peak memory. By default it uses a tiny random RoBERTa and synthetic functions, so it
needs no downloads; this measures the training loop, not model quality.

    python scripts/bench_training.py                      # tiny fixture, CPU or CUDA
    python scripts/bench_training.py --base-model microsoft/codebert-base --device cuda

Each configuration runs in its own process so the peak-memory numbers do not mix.
"""

import argparse
import json
import multiprocessing
import random
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CONFIGS = {
    # What the 66-minute T4 run used: one row per step, fp32, random batches.
    "old": dict(batch_size=1, accumulation=8, precision="fp32", group_by_length=False, tuned_loader=False),
    # New defaults: real batches, mixed precision on CUDA, length-grouped batches.
    "new": dict(batch_size=16, accumulation=1, precision="auto", group_by_length=True, tuned_loader=True),
}
WORDS = ["def", "return", "if", "else", "for", "in", "x", "y", "i", "n", "+", "-", "*", "/",
         "==", "!=", "<", ">", "and", "or", "(", ")", ":", "0", "1", "2", "len", "range"]


def tiny_fixture(folder):
    """A small random RoBERTa + word-level tokenizer; no pretrained weights involved."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast, RobertaConfig, RobertaForSequenceClassification

    vocab = {token: i for i, token in enumerate(["<pad>", "<s>", "</s>", "<unk>", *WORDS])}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>", special_tokens=[("<s>", 1), ("</s>", 2)]
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="<pad>", bos_token="<s>", eos_token="</s>",
        unk_token="<unk>", model_max_length=512,
    )
    fast.model_input_names = ["input_ids", "attention_mask"]
    fast.save_pretrained(folder)
    RobertaForSequenceClassification(
        RobertaConfig(
            vocab_size=len(vocab), hidden_size=128, num_hidden_layers=2, num_attention_heads=4,
            intermediate_size=256, max_position_embeddings=514, pad_token_id=0, num_labels=2,
        )
    ).save_pretrained(folder)


def synthetic_rows(count, seed=42):
    """Functions with a long-tailed length distribution, like real code."""
    rng = random.Random(seed)
    rows = []
    for _ in range(count):
        length = min(600, max(6, int(rng.lognormvariate(4.3, 0.8))))
        rows.append((" ".join(rng.choice(WORDS) for _ in range(length)), rng.randint(0, 1)))
    return rows


def run(name, settings, args, queue):
    import resource

    import torch
    from datasets import Dataset
    from src.bug_detection.lora_common import (
        add_classifier_lora,
        hardware_options,
        loader_options,
        pad_multiple,
        precision_name,
        tokenize_once,
    )
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    torch.set_num_threads(args.threads)
    set_seed(42)
    options = hardware_options(args.device, settings["precision"])
    cuda = not options["use_cpu"]
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model, _ = add_classifier_lora(
        AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=2)
    )
    rows = synthetic_rows(args.rows)
    features, _ = tokenize_once(tokenizer, [code for code, _ in rows], args.max_length)
    data = Dataset.from_dict({**features, "labels": [label for _, label in rows]})
    loader = loader_options(options) if settings["tuned_loader"] else {"dataloader_num_workers": 0}
    with tempfile.TemporaryDirectory() as scratch:
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=scratch,
                max_steps=args.steps,
                per_device_train_batch_size=settings["batch_size"],
                gradient_accumulation_steps=settings["accumulation"],
                learning_rate=2e-4,
                group_by_length=settings["group_by_length"],
                save_strategy="no",
                eval_strategy="no",
                logging_strategy="no",
                report_to="none",
                seed=42,
                disable_tqdm=True,
                **loader,
                **options,
            ),
            train_dataset=data,
            data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=pad_multiple(options)),
        )
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        trainer.train()
        if cuda:
            torch.cuda.synchronize()
        seconds = time.perf_counter() - started
    samples = args.steps * settings["batch_size"] * settings["accumulation"]
    queue.put(
        {
            "config": name,
            "device": "cuda" if cuda else "cpu",
            "precision": precision_name(options),
            "batch_size": settings["batch_size"],
            "accumulation": settings["accumulation"],
            "steps": args.steps,
            "seconds": round(seconds, 2),
            "steps_per_sec": round(args.steps / seconds, 2),
            "samples_per_sec": round(samples / seconds, 1),
            "peak_memory_mb": round(
                torch.cuda.max_memory_allocated() / 2**20
                if cuda
                else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,  # KiB on Linux
                1,
            ),
            "memory_kind": "GPU allocated" if cuda else "process RSS",
        }
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", default=None, help="default: a tiny random RoBERTa fixture")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--rows", type=int, default=2000)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--threads", type=int, default=4, help="CPU threads for torch")
    p.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    args = p.parse_args()

    context = multiprocessing.get_context("spawn")
    with tempfile.TemporaryDirectory() as fixture:
        if args.base_model is None:
            tiny_fixture(fixture)
            args.base_model = fixture
            print("Model: tiny random RoBERTa fixture (loop timing only, not model quality)")
        else:
            print(f"Model: {args.base_model}")
        results = []
        for name in args.configs:
            queue = context.Queue()
            worker = context.Process(target=run, args=(name, CONFIGS[name], args, queue))
            worker.start()
            worker.join()
            if worker.exitcode != 0:
                raise SystemExit(f"configuration {name!r} failed (exit code {worker.exitcode})")
            results.append(queue.get())

    header = f"{'config':<7}{'device':<7}{'prec':<6}{'batch':>6}{'accum':>6}{'steps/s':>9}{'samples/s':>11}{'peak MB':>9}"
    print(header + "\n" + "-" * len(header))
    for r in results:
        print(
            f"{r['config']:<7}{r['device']:<7}{r['precision']:<6}{r['batch_size']:>6}{r['accumulation']:>6}"
            f"{r['steps_per_sec']:>9}{r['samples_per_sec']:>11}{r['peak_memory_mb']:>9}"
        )
    by_name = {r["config"]: r for r in results}
    if {"old", "new"} <= set(by_name):
        speedup = by_name["new"]["samples_per_sec"] / by_name["old"]["samples_per_sec"]
        print(f"\nThroughput: {speedup:.1f}x samples/sec (new vs old). Peak memory: {results[0]['memory_kind']}.")
        print("A 'step' is one optimizer update: old = 8 rows (1 x 8 accumulated), new = 16 rows.")
    print(json.dumps(results))


if __name__ == "__main__":
    main()
