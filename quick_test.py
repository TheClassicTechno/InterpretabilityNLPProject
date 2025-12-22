#!/usr/bin/env python3

import torch
from src.model_loader import ModelLoader
from src.baseline import generate_response, get_model_accuracy
from src.dataset import SimpleDataset
import json
from pathlib import Path

print("Quick test of interpretability pipeline")
print("="*60)

print("\n[1] Loading model...")
loader = ModelLoader("gpt2")
loader.load()
model = loader.get_model()
tokenizer = loader.get_tokenizer()
device = loader.get_device()

print(f"Model: gpt2 on {device}")
print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

print("\n[2] Loading dataset...")
questions, answers = SimpleDataset.get_arithmetic_tasks()
print(f"Questions: {len(questions)}")
for i, (q, a) in enumerate(zip(questions[:3], answers[:3]), 1):
    print(f"  {i}. Q: {q} → A: {a}")

print("\n[3] Testing baseline generation...")
test_q = questions[0]
test_a = answers[0]
response = generate_response(model, tokenizer, test_q, device=device)
print(f"Q: {test_q}")
print(f"Expected: {test_a}")
print(f"Got: {response}")
print(f"Correct: {str(test_a).strip() in response}")

print("\n[4] Measuring baseline accuracy...")
acc = get_model_accuracy(model, tokenizer, questions, answers, device=device)
print(f"Baseline accuracy: {acc:.1%}")

print("\n[5] Model config...")
print(f"  Num layers: {model.config.num_hidden_layers}")
print(f"  Num heads: {model.config.num_attention_heads}")
print(f"  Hidden size: {model.config.hidden_size}")

print("\nBasic test completed successfully!")
print("="*60)
