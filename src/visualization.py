import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple


class Visualizer:
    """Create plots for interpretability analysis."""
    
    @staticmethod
    def plot_head_importance(head_importance: Dict, num_layers: int, num_heads: int, output_path=None):
        """
        Plot attention head importance heatmap.
        
        Args:
            head_importance: Dictionary mapping (layer, head) to importance score.
            num_layers: Number of layers in model.
            num_heads: Number of attention heads per layer.
            output_path: Path to save figure.
        """
        importance_matrix = np.zeros((num_layers, num_heads))
        
        for (layer_idx, head_idx), importance in head_importance.items():
            if layer_idx < num_layers and head_idx < num_heads:
                importance_matrix[layer_idx, head_idx] = importance
        
        plt.figure(figsize=(12, 6))
        plt.imshow(importance_matrix, cmap="RdYlBu_r", aspect="auto")
        plt.colorbar(label="Importance Score")
        plt.xlabel("Attention Head")
        plt.ylabel("Layer")
        plt.title("Attention Head Importance Heatmap")
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    
    @staticmethod
    def plot_layer_importance(layer_importance: Dict, output_path=None):
        """
        Plot layer importance as bar chart.
        
        Args:
            layer_importance: Dictionary mapping layer index to importance.
            output_path: Path to save figure.
        """
        layers = sorted(layer_importance.keys())
        importances = [layer_importance[l] for l in layers]
        
        plt.figure(figsize=(10, 5))
        plt.bar(layers, importances, color="steelblue")
        plt.xlabel("Layer Index")
        plt.ylabel("Importance Score")
        plt.title("Layer Importance for Reasoning")
        plt.grid(axis="y", alpha=0.3)
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    
    @staticmethod
    def plot_accuracy_vs_ablation(ablation_results: Dict, output_path=None):
        """
        Plot accuracy drop after ablating different components.
        
        Args:
            ablation_results: Dictionary with component names and accuracy scores.
            output_path: Path to save figure.
        """
        components = list(ablation_results.keys())
        accuracies = [ablation_results[c] for c in components]
        
        plt.figure(figsize=(10, 5))
        bars = plt.bar(components, accuracies, color="coral")
        
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.2f}',
                    ha='center', va='bottom')
        
        plt.ylabel("Accuracy")
        plt.title("Accuracy After Component Ablation")
        plt.ylim(0, 1)
        plt.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=45, ha="right")
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    
    @staticmethod
    def plot_failure_distribution(failure_analysis: Dict, output_path=None):
        """
        Plot distribution of failure types.
        
        Args:
            failure_analysis: Dictionary with failure counts by type.
            output_path: Path to save figure.
        """
        patterns = failure_analysis.get("pattern_counts", {})
        
        if not patterns:
            return
        
        labels = list(patterns.keys())
        counts = list(patterns.values())
        
        plt.figure(figsize=(8, 6))
        colors = ["#ff9999", "#ffcc99", "#99ccff", "#99ff99"]
        plt.pie(counts, labels=labels, autopct='%1.1f%%', colors=colors)
        plt.title("Distribution of Error Types")
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    
    @staticmethod
    def plot_activation_comparison(stats: Dict, output_path=None):
        """
        Plot comparison of activations for correct vs incorrect reasoning.
        
        Args:
            stats: Dictionary with activation statistics.
            output_path: Path to save figure.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        correct_mean = stats.get("correct_mean", 0)
        incorrect_mean = stats.get("incorrect_mean", 0)
        correct_std = stats.get("correct_std", 0)
        incorrect_std = stats.get("incorrect_std", 0)
        
        means = [correct_mean, incorrect_mean]
        stds = [correct_std, incorrect_std]
        labels = ["Correct", "Incorrect"]
        
        axes[0].bar(labels, means, yerr=stds, capsize=5, color=["green", "red"], alpha=0.7)
        axes[0].set_ylabel("Mean Activation")
        axes[0].set_title("Activation Mean Comparison")
        axes[0].grid(axis="y", alpha=0.3)
        
        cosine_sim = stats.get("cosine_similarity", 0)
        l2_dist = stats.get("l2_distance", 0)
        
        axes[1].bar(["Cosine Sim", "L2 Distance"], [cosine_sim, l2_dist], color=["blue", "orange"], alpha=0.7)
        axes[1].set_ylabel("Score")
        axes[1].set_title("Activation Similarity Metrics")
        axes[1].grid(axis="y", alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
