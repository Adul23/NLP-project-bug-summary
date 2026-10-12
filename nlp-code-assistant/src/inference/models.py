"""Local artifact loading. Transformer downloads are never triggered by inference."""
import json
from pathlib import Path

import joblib

from src.bug_detection.split_dataset import ROOT


class TfidfPredictor:
    def __init__(self, folder):
        folder = Path(folder)
        self.vectorizer = joblib.load(folder / 'tfidf_vectorizer.joblib')
        self.model = joblib.load(folder / 'logistic_regression.joblib')
        if len(self.vectorizer.vocabulary_) != self.model.n_features_in_:
            raise ValueError('Vectorizer and classifier feature dimensions do not match')
        if list(self.model.classes_) != [0, 1]:
            raise ValueError('Artifact label mapping must be 0=CLEAN, 1=BUGGY')

    def predict(self, code):
        features = self.vectorizer.transform([code])
        probability = float(self.model.predict_proba(features)[0, 1])
        label = int(self.model.predict(features)[0])
        return {'label': 'BUGGY' if label else 'CLEAN', 'bug_probability': probability,
                'confidence': probability if label else 1 - probability, 'model': 'tfidf'}


class LoraPredictor:
    def __init__(self, folder, *, summarizer=False, device='cpu'):
        import torch
        from peft import PeftModel
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
        folder = Path(folder)
        config = json.loads((folder / 'training_config.json').read_text())
        if config.get('status') != 'trained':
            raise ValueError('Adapter has not completed training')
        self.config = config
        self.summarizer = summarizer
        self.tokenizer = AutoTokenizer.from_pretrained(folder / 'adapter', local_files_only=True)
        cls = AutoModelForSeq2SeqLM if summarizer else AutoModelForSequenceClassification
        kwargs = {} if summarizer else {'num_labels': 2, 'id2label': {0: 'CLEAN', 1: 'BUGGY'}, 'label2id': {'CLEAN': 0, 'BUGGY': 1}}
        base = cls.from_pretrained(config['base_model'], revision=config['revision'], local_files_only=True, **kwargs)
        self.model = PeftModel.from_pretrained(base, folder / 'adapter', local_files_only=True).to(device).eval()
        self.torch = torch

    def predict(self, code):
        if self.summarizer:
            raise ValueError('Use generate for a summarizer')
        inputs = self.tokenizer(code, return_tensors='pt', truncation=True,
                                max_length=self.config['max_length'], return_token_type_ids=False).to(self.model.device)
        with self.torch.inference_mode():
            probabilities = self.model(**inputs).logits.softmax(-1)[0]
        label = int(probabilities.argmax())
        p = float(probabilities[1])
        return {'label': 'BUGGY' if label else 'CLEAN', 'bug_probability': p,
                'confidence': p if label else 1-p, 'model': 'codebert'}

    def generate(self, code):
        if not self.summarizer:
            raise ValueError('Not a summarizer')
        from src.summarization.train_lora import PROMPT
        inputs = self.tokenizer(PROMPT + code, return_tensors='pt', truncation=True,
                                max_length=self.config['max_length'], return_token_type_ids=False).to(self.model.device)
        with self.torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
        return {'explanation': self.tokenizer.decode(ids[0], skip_special_tokens=True),
                'explanation_source': 'lora_generated_unverified', 'verified': False,
                'bug_type': None, 'suggested_fix': None}
