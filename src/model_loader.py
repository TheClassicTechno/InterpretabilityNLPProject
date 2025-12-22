import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelLoader:
    """Load and manage a pretrained language model."""
    
    def __init__(self, model_name, device=None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
    
    def load(self):
        """Load model and tokenizer."""
        print(f"Loading {self.model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        if self.device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=torch.float16,
                device_map="auto",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=torch.float32,
            )
            self.model = self.model.to(self.device)
        
        self.model.eval()
        return self
    
    def get_model(self):
        """Return the loaded model."""
        return self.model
    
    def get_tokenizer(self):
        """Return the tokenizer."""
        return self.tokenizer
    
    def get_device(self):
        """Return the device."""
        return self.device
