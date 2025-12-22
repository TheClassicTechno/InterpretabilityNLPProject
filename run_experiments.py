#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path


def run_command(cmd):
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print('='*60)
    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    """Run a series of standard experiments."""
    
    print("\nLLM Interpretability Experiment Suite")
    print("="*60)
    
    base_cmd = [sys.executable, "src/experiment.py"]
    
    experiments = [
        {
            "name": "Arithmetic (Day 1)",
            "args": ["--model", "gpt2", "--dataset", "arithmetic", "--output", "results/day1_arithmetic"]
        },
        {
            "name": "Multi-step Reasoning (Day 2a)",
            "args": ["--model", "gpt2", "--dataset", "multistep", "--output", "results/day2_multistep"]
        },
        {
            "name": "Logic Tasks (Day 2b)",
            "args": ["--model", "gpt2", "--dataset", "logic", "--output", "results/day2_logic"]
        },
        {
            "name": "GSM8K Math (Day 3)",
            "args": ["--model", "gpt2", "--dataset", "gsm8k", "--output", "results/day3_gsm8k"]
        },
    ]
    
    results = {}
    
    for exp in experiments:
        cmd = base_cmd + exp["args"]
        success = run_command(cmd)
        results[exp["name"]] = "PASSED" if success else "FAILED"
        
        if not success:
            print(f"\nWarning: {exp['name']} failed")
            user_input = input("Continue with next experiment? (y/n): ")
            if user_input.lower() != "y":
                break
    
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    for name, status in results.items():
        symbol = "[PASS]" if status == "PASSED" else "[FAIL]"
        print(f"{symbol} {name}: {status}")
    
    print("\nResults saved in results/ directory")
    print("Check each subdirectory for JSON and PNG outputs")


if __name__ == "__main__":
    main()
