#!/usr/bin/env python3

import json
import sys
from pathlib import Path


def load_results(results_file):
    """Load results from JSON file."""
    with open(results_file, "r") as f:
        return json.load(f)


def print_summary(results):
    """Print a formatted summary of results."""
    
    print("\n" + "="*70)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*70)
    
    print(f"\nModel: {results.get('model_name', 'Unknown')}")
    print(f"Dataset: {results.get('dataset_type', 'Unknown')} ({results.get('num_samples', 0)} samples)")
    print(f"Baseline Accuracy: {results.get('baseline_accuracy', 0):.2%}")
    
    print("\n" + "-"*70)
    print("TOP IMPORTANT HEADS")
    print("-"*70)
    
    if "head_analysis" in results and "top_heads" in results["head_analysis"]:
        for i, head in enumerate(results["head_analysis"]["top_heads"][:5], 1):
            print(f"{i}. Layer {head['layer']}, Head {head['head']}: {head['importance']:.6f}")
    else:
        print("No head analysis available")
    
    print("\n" + "-"*70)
    print("LAYER IMPORTANCE")
    print("-"*70)
    
    if "layer_analysis" in results and "layer_importance" in results["layer_analysis"]:
        layer_imp = results["layer_analysis"]["layer_importance"]
        
        sorted_layers = sorted(
            [(int(k) if k != "baseline_accuracy" else -1, float(v))
             for k, v in layer_imp.items() if k != "baseline_accuracy"],
            key=lambda x: x[1],
            reverse=True
        )
        
        for layer_idx, importance in sorted_layers[:5]:
            print(f"Layer {layer_idx}: {importance:.6f}")
    else:
        print("No layer analysis available")
    
    print("\n" + "-"*70)
    print("ERROR BREAKDOWN")
    print("-"*70)
    
    if "error_patterns" in results:
        patterns = results["error_patterns"]
        total = sum(patterns.values())
        for error_type, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total * 100) if total > 0 else 0
            print(f"{error_type:.<25} {count:>3} ({pct:>5.1f}%)")
    else:
        print("No error analysis available")
    
    print("\n" + "="*70 + "\n")


def main():
    """Inspect results from command line."""
    
    if len(sys.argv) < 2:
        print("Usage: python inspect_results.py <path_to_results.json>")
        print("\nExamples:")
        print("  python inspect_results.py results/day1_arithmetic/results.json")
        print("  python inspect_results.py results/day2_multistep/results.json")
        sys.exit(1)
    
    results_file = Path(sys.argv[1])
    
    if not results_file.exists():
        print(f"Error: File not found: {results_file}")
        sys.exit(1)
    
    try:
        results = load_results(results_file)
        print_summary(results)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
