import json
import argparse
from pathlib import Path

from src.model_loader import ModelLoader
from src.baseline import get_model_accuracy
from src.ablation import HeadAblator
from src.analysis import LayerwiseAnalyzer, FailureAnalyzer
from src.visualization import Visualizer
from src.dataset import SimpleDataset


def run_experiment(model_name, dataset_type, output_dir):
    """
    Run full interpretability experiment.
    
    Args:
        model_name: HuggingFace model identifier.
        dataset_type: Type of dataset (arithmetic, multistep, logic, gsm8k).
        output_dir: Directory to save results.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("[1/5] Loading model...")
    loader = ModelLoader(model_name)
    loader.load()
    model = loader.get_model()
    tokenizer = loader.get_tokenizer()
    device = loader.get_device()
    
    print("[2/5] Loading dataset...")
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
    
    print("[3/5] Computing baseline accuracy...")
    baseline_acc = get_model_accuracy(model, tokenizer, questions, answers, device)
    print(f"Baseline accuracy: {baseline_acc:.2%}")
    
    results = {
        "model_name": model_name,
        "dataset_type": dataset_type,
        "num_samples": len(questions),
        "baseline_accuracy": float(baseline_acc),
    }
    
    print("[4/5] Running ablation analysis...")
    ablator = HeadAblator(model, tokenizer, device)
    head_rankings = ablator.rank_heads(questions, answers)
    results["head_analysis"] = {
        "baseline_accuracy": head_rankings["baseline_accuracy"],
        "top_heads": [
            {"layer": h[0][0], "head": h[0][1], "importance": float(h[1])}
            for h in head_rankings["sorted_heads"][:10]
        ],
    }
    
    print("[5/5] Running layer analysis...")
    analyzer = LayerwiseAnalyzer(model, tokenizer, device)
    layer_imp = analyzer.get_layer_importance(questions, answers)
    results["layer_analysis"] = {
        "baseline_accuracy": layer_imp["baseline_accuracy"],
        "layer_importance": {
            str(k): float(v) for k, v in layer_imp["layer_importance"].items()
        },
    }
    
    print("\n[VISUALIZATION] Creating plots...")
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    
    Visualizer.plot_head_importance(
        head_rankings["head_importance"],
        num_layers,
        num_heads,
        output_path / "head_importance.png"
    )
    
    Visualizer.plot_layer_importance(
        layer_imp["layer_importance"],
        output_path / "layer_importance.png"
    )
    
    print("\n[FAILURE ANALYSIS] Analyzing errors...")
    failure_analyzer = FailureAnalyzer(model, tokenizer, device)
    failure_analysis = failure_analyzer.categorize_failures(questions, answers)
    results["failure_analysis"] = {
        "total": failure_analysis["total"],
        "correct_count": failure_analysis["correct_count"],
        "incorrect_count": failure_analysis["incorrect_count"],
        "accuracy": float(failure_analysis["accuracy"]),
    }
    
    error_patterns = failure_analyzer.analyze_error_patterns(
        failure_analysis["incorrect_examples"]
    )
    results["error_patterns"] = error_patterns["pattern_counts"]
    
    Visualizer.plot_failure_distribution(
        error_patterns,
        output_path / "failure_distribution.png"
    )
    
    print("\n[RESULTS SUMMARY]")
    print(f"Baseline accuracy: {results['baseline_accuracy']:.2%}")
    print(f"Top 3 important heads:")
    for i, head in enumerate(results["head_analysis"]["top_heads"][:3], 1):
        print(f"  {i}. Layer {head['layer']}, Head {head['head']}: {head['importance']:.4f}")
    
    print(f"\nError breakdown:")
    for error_type, count in results["error_patterns"].items():
        print(f"  {error_type}: {count}")
    
    results_file = output_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    print(f"Plots saved to {output_path}/")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LLM interpretability experiments"
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
    
    run_experiment(args.model, args.dataset, args.output)
