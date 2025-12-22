import json
import argparse
from pathlib import Path

from src.model_loader import ModelLoader
from src.baseline import get_model_accuracy, generate_response
from src.analysis import FailureAnalyzer
from src.visualization import Visualizer
from src.dataset import SimpleDataset


def run_lightweight_experiment(model_name, dataset_type, output_dir):
    """
    Run lightweight version focusing on baseline and failure analysis.
    Full head ablation is computationally expensive and skipped here.
    
    Args:
        model_name: HuggingFace model identifier.
        dataset_type: Type of dataset.
        output_dir: Directory to save results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("[1/3] Loading model...")
    loader = ModelLoader(model_name)
    loader.load()
    model = loader.get_model()
    tokenizer = loader.get_tokenizer()
    device = loader.get_device()
    
    print(f"Model loaded: {model_name} on {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    print("\n[2/3] Loading dataset...")
    if dataset_type == "arithmetic":
        questions, answers = SimpleDataset.get_arithmetic_tasks()
    elif dataset_type == "multistep":
        questions, answers = SimpleDataset.get_multistep_tasks()
    elif dataset_type == "logic":
        questions, answers = SimpleDataset.get_logic_tasks()
    elif dataset_type == "gsm8k":
        questions, answers = SimpleDataset.load_gsm8k_sample(num_samples=5)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    print(f"Dataset: {len(questions)} samples")
    for i, (q, a) in enumerate(zip(questions[:2], answers[:2]), 1):
        print(f"  {i}. {q[:50]}... → {a}")
    
    print("\n[3/3] Measuring baseline accuracy...")
    baseline_acc = get_model_accuracy(model, tokenizer, questions, answers, device)
    print(f"Baseline accuracy: {baseline_acc:.1%}")
    
    print("\nRunning failure analysis...")
    failure_analyzer = FailureAnalyzer(model, tokenizer, device)
    failure_analysis = failure_analyzer.categorize_failures(questions, answers)
    
    print(f"Correct: {failure_analysis['correct_count']}/{len(questions)}")
    print(f"Incorrect: {failure_analysis['incorrect_count']}/{len(questions)}")
    
    error_patterns = failure_analyzer.analyze_error_patterns(
        failure_analysis["incorrect_examples"]
    )
    
    results = {
        "model_name": model_name,
        "dataset_type": dataset_type,
        "num_samples": len(questions),
        "baseline_accuracy": float(baseline_acc),
        "model_params_millions": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
        "num_layers": model.config.num_hidden_layers,
        "num_heads": model.config.num_attention_heads,
        "failure_analysis": {
            "total": failure_analysis["total"],
            "correct_count": failure_analysis["correct_count"],
            "incorrect_count": failure_analysis["incorrect_count"],
            "accuracy": float(failure_analysis["accuracy"]),
        },
        "error_patterns": error_patterns["pattern_counts"],
        "examples": {
            "correct": [
                {"question": ex["question"], "answer": ex["answer"]}
                for ex in failure_analysis["correct_examples"][:2]
            ],
            "incorrect": [
                {"question": ex["question"], "answer": ex["answer"], "response": ex["response"]}
                for ex in failure_analysis["incorrect_examples"][:2]
            ]
        }
    }
    
    print("\n[VISUALIZATION] Creating plots...")
    Visualizer.plot_failure_distribution(
        error_patterns,
        output_path / "failure_distribution.png"
    )
    print(f"Plot saved: failure_distribution.png")
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_type} ({len(questions)} samples)")
    print(f"Baseline Accuracy: {baseline_acc:.1%}")
    print(f"\nError Breakdown:")
    for error_type, count in error_patterns["pattern_counts"].items():
        pct = count / len(failure_analysis["incorrect_examples"]) * 100 if failure_analysis["incorrect_examples"] else 0
        print(f"  {error_type:.<30} {count} ({pct:>5.1f}%)")
    
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    print(f"Plots saved to {output_path}/")
    print("="*70)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run lightweight LLM interpretability analysis"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="arithmetic",
        choices=["arithmetic", "multistep", "logic", "gsm8k"],
        help="Dataset type to use",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory for results",
    )
    
    args = parser.parse_args()
    
    run_lightweight_experiment(args.model, args.dataset, args.output)
