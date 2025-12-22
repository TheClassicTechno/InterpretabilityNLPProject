import torch
import torch.nn as nn
from typing import Dict, List, Tuple


class HeadAblator:
    """Ablate attention heads and measure performance impact."""
    
    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.num_layers = model.config.num_hidden_layers
        self.num_heads = model.config.num_attention_heads
    
    def zero_out_head(self, layer_idx, head_idx):
        """Zero out a specific attention head."""
        attn_layer = self.model.transformer.h[layer_idx].attn
        attn_layer.register_forward_hook(
            lambda module, input, output: self._hook_zero_head(
                output, head_idx, self.num_heads
            )
        )
    
    @staticmethod
    def _hook_zero_head(output, head_idx, num_heads):
        """Hook to zero out a head in attention output."""
        attn_output = output[0]
        head_dim = attn_output.shape[-1] // num_heads
        start = head_idx * head_dim
        end = start + head_dim
        attn_output[..., start:end] = 0
        return (attn_output, *output[1:])
    
    def ablate_head(self, question, answer, layer_idx, head_idx):
        """
        Ablate a single head and measure accuracy impact.
        
        Args:
            question: Input question text.
            answer: Correct answer text.
            layer_idx: Layer index (0 to num_layers-1).
            head_idx: Head index (0 to num_heads-1).
        
        Returns:
            Boolean indicating if the answer is correct after ablation.
        """
        hooks = []
        
        def zero_hook(module, input, output):
            attn_output = output[0].clone()
            head_dim = attn_output.shape[-1] // self.num_heads
            start = head_idx * head_dim
            end = start + head_dim
            attn_output[..., start:end] = 0
            return (attn_output, *output[1:])
        
        attn_layer = self.model.transformer.h[layer_idx].attn
        hook = attn_layer.register_forward_hook(zero_hook)
        hooks.append(hook)
        
        try:
            inputs = self.tokenizer(question, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            is_correct = str(answer).strip() in response
            
            return is_correct
        finally:
            for h in hooks:
                h.remove()
    
    def ablate_layer(self, question, answer, layer_idx):
        """
        Zero out an entire layer and measure impact.
        
        Args:
            question: Input question text.
            answer: Correct answer text.
            layer_idx: Layer index to ablate.
        
        Returns:
            Boolean indicating if answer is correct after ablation.
        """
        hooks = []
        
        def zero_layer_hook(module, input, output):
            return (torch.zeros_like(output[0]), *output[1:])
        
        attn_layer = self.model.transformer.h[layer_idx].attn
        hook = attn_layer.register_forward_hook(zero_layer_hook)
        hooks.append(hook)
        
        try:
            inputs = self.tokenizer(question, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            is_correct = str(answer).strip() in response
            
            return is_correct
        finally:
            for h in hooks:
                h.remove()
    
    def rank_heads(self, questions: List[str], answers: List[str]) -> Dict:
        """
        Rank attention heads by importance.
        
        Args:
            questions: List of question texts.
            answers: List of answer texts.
        
        Returns:
            Dictionary with head rankings and importance scores.
        """
        baseline_correct = sum(
            1 for q, a in zip(questions, answers)
            if self._test_answer(q, a)
        )
        baseline_acc = baseline_correct / len(questions)
        
        head_importance = {}
        
        for layer_idx in range(self.num_layers):
            for head_idx in range(self.num_heads):
                correct = 0
                for q, a in zip(questions, answers):
                    if self.ablate_head(q, a, layer_idx, head_idx):
                        correct += 1
                
                acc_with_ablation = correct / len(questions)
                importance = baseline_acc - acc_with_ablation
                head_importance[(layer_idx, head_idx)] = importance
        
        sorted_heads = sorted(
            head_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            "baseline_accuracy": baseline_acc,
            "head_importance": head_importance,
            "sorted_heads": sorted_heads,
        }
    
    def _test_answer(self, question, answer):
        """Test if model produces correct answer."""
        inputs = self.tokenizer(question, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return str(answer).strip() in response
