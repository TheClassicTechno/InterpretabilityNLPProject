# LLM NLP Interpretability Project

personal project by juli. explored how language models (LLMs) solve reasoning problems by analyzing which parts of the model actually do the work.

## What This Does

Instead of treating models as black boxes, I systematically break them down into parts:
1. Test baseline accuracy on math and logic problems
2. Zero out attention heads one at a time and measure performance drops
3. Identify which heads matter most
4. Analyze what kinds of errors the model makes

This tells us which parts of the model are responsible for reasoning and where failures occur.

## Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run fast test (2-3 minutes):
   ```bash
   python -m src.experiment_lite --model gpt2 --dataset arithmetic --output results/test
   ```

3. View results:
   ```bash
   python inspect_results.py results/test/results.json
   ```

## How It Works

### Step 1: Load Model
Load any HuggingFace model (gpt2, mistral, llama, etc).

### Step 2: Test Baseline
Generate responses to questions and measure accuracy.

Example questions:
- "What is 5 + 3?" (expected: 8)
- "Alice has 10 dollars, spends 4. How much left?" (expected: 6)
- "All dogs are animals. Max is a dog. Is Max an animal?" (expected: Yes)

### Step 3: Run Ablations
For each attention head in the model:
- Zero it out
- Test accuracy again
- Measure how much performance dropped
- Rank heads by importance

### Step 4: Analyze Failures
Categorize errors made by the model:
- Hallucination: generates plausible but wrong text
- Wrong operation: uses wrong math operation
- Off-by-one: answer is close but not exact
- Incomplete: answer is cut off

### Step 5: Generate Visualizations
Create plots showing:
- Which heads matter most (heatmap)
- Which layers are important (bar chart)
- What errors happen most (pie chart)

## Datasets Made

**Arithmetic** (8 questions)
- Simple math: 5+3, 10-4, 6*2, etc.

**Multi-step** (5 questions)  
- Word problems requiring multiple operations

**Logic** (4 questions)
- Logical inference: "All X are Y, Z is X, is Z Y?"

**GSM8K** (5 questions)
- Real grade school math problems

## Commands

Run lightweight version (fast, 2-3 minutes):
```bash
python -m src.experiment_lite --model gpt2 --dataset arithmetic --output results/test
```

Run full version (includes head ablation, slower):
```bash
python -m src.experiment --model gpt2 --dataset arithmetic --output results/full
```

Test different dataset:
```bash
python -m src.experiment_lite --model gpt2 --dataset logic --output results/logic
```

Test different model:
```bash
python -m src.experiment_lite --model distilgpt2 --dataset arithmetic --output results/distil
```

## What I Found From My Experiments

Small language models (GPT-2, DistilGPT2) fail at basic reasoning tasks because they're trained to predict the next word, not solve math or logic problems. 

**The Results**
- Arithmetic: 0% accuracy (models just hallucinate numbers)
- Logic: 50% on simple deduction, then fails on harder problems
- GSM8K: 0% accuracy (too complex for models never trained on reasoning)

When I turned off individual attention heads, nothing changed the 0% baseline. This revealed that the models lack the fundamental capability to reason, not just weak components. No single head is responsible, since the whole architecture wasn't designed for this task.

Hallucination is the dominant failure mode (75-100% of errors). The models confidently generate plausible-sounding but completely wrong answers instead of saying "I don't know." This happened the same way in both GPT-2 and DistilGPT2, showing it's a property of how these models work, not a size issue.

**What This Means**
- Small models fail at reasoning consistently, independent of architecture tweaks
- To solve reasoning, you need models trained on step-by-step solutions (like Mistral or LLaMA trained on math)
- Only having more parameters doesn't help—training data and methodology matter more

In conclusion, I found why and how small models fail at math.

View results:
```bash
python inspect_results.py results/test/results.json
```