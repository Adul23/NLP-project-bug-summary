"""Predict from a UTF-8 file or stdin without executing source code."""
import argparse
import json
import sys
from pathlib import Path
from src.bug_detection.split_dataset import ROOT
from .models import TfidfPredictor, LoraPredictor
from src.summarization.prepare import verified_change


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--file', type=Path, help='Defaults to stdin')
    p.add_argument('--model', choices=['tfidf', 'codebert', 'summarizer'], default='tfidf')
    p.add_argument('--model-dir', type=Path)
    p.add_argument('--original-file', type=Path, help='Verify a synthetic change against a known original')
    a = p.parse_args()
    code = a.file.read_text() if a.file else sys.stdin.read()
    if not code.strip() or len(code) > 20000:
        p.error('Code must contain 1–20000 characters')
    folder = a.model_dir or ROOT / 'models' / {'tfidf':'tfidf_v2','codebert':'codebert_lora','summarizer':'codet5_lora'}[a.model]
    try:
        predictor = TfidfPredictor(folder) if a.model == 'tfidf' else LoraPredictor(folder, summarizer=a.model == 'summarizer')
        result = predictor.generate(code) if a.model == 'summarizer' else predictor.predict(code)
        if a.original_file:
            result['verified_change'] = verified_change(a.original_file.read_text(), code)
    except (OSError, ValueError, ImportError) as exc:
        p.exit(1, f'Model unavailable or invalid input: {exc}\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
