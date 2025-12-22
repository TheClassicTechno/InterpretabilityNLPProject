import json
from pathlib import Path


class Config:
    """Configuration for interpretability experiments."""
    
    # Model settings
    DEFAULT_MODEL = "gpt2"
    DEVICE = "auto"
    
    # Generation settings
    MAX_TOKENS = 50
    TEMPERATURE = 0.7
    
    # Ablation settings
    SAMPLE_SIZE_PER_TASK = 10
    
    # Dataset settings
    DATASETS = {
        "arithmetic": {
            "name": "Simple Arithmetic",
            "description": "Basic arithmetic operations",
        },
        "multistep": {
            "name": "Multi-step Reasoning",
            "description": "Reasoning requiring multiple steps",
        },
        "logic": {
            "name": "Logic Tasks",
            "description": "Logical reasoning and inference",
        },
        "gsm8k": {
            "name": "GSM8K Math",
            "description": "Grade school math problems",
        },
    }
    
    # Output settings
    OUTPUT_DIR = "results"
    
    # Visualization settings
    PLOT_DPI = 150
    PLOT_FORMAT = "png"
    
    # Report settings
    REPORT_TOP_N_HEADS = 10
    
    @staticmethod
    def get_device():
        """Get device to use."""
        import torch
        if Config.DEVICE == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return Config.DEVICE
    
    @staticmethod
    def get_model_config(model_name):
        """Get model-specific configuration."""
        return {
            "model_name": model_name,
            "max_tokens": Config.MAX_TOKENS,
            "temperature": Config.TEMPERATURE,
            "device": Config.get_device(),
        }
    
    @staticmethod
    def create_output_dir(output_path):
        """Create output directory if it doesn't exist."""
        Path(output_path).mkdir(parents=True, exist_ok=True)
        return output_path
