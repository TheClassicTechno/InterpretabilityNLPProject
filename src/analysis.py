import torch
import numpy as np
from typing import Dict, List, Tuple


class LayerwiseAnalyzer:
    """Analyze reasoning emergence across layers."""
    
    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.num_layers = model.config.num_hidden_layers
    
    def get_layer_importance(self, questions: List[str], answers: List[str]) -> Dict:
        """
        Measure which layers are critical for correct reasoning.
        
        Args:
            questions: List of question texts.
            answers: List of answer texts.
        
        Returns:
            Dictionary with layer importance scores.
        """
        baseline_correct = sum(
            1 for q, a in zip(questions, answers)
            if self._test_answer(q, a)
        )
        baseline_acc = baseline_correct / len(questions) if questions else 0.0
        
        layer_importance = {}
        
        for layer_idx in range(self.num_layers):
            hooks = []
            
            def zero_layer_hook(module, input, output, layer=layer_idx):
                return (torch.zeros_like(output[0]), *output[1:])
            
            attn_layer = self.model.transformer.h[layer_idx].attn
            hook = attn_layer.register_forward_hook(zero_layer_hook)
            hooks.append(hook)
            
            correct = 0
            for q, a in zip(questions, answers):
                if self._test_answer(q, a):
                    correct += 1
            
            acc_with_ablation = correct / len(questions) if questions else 0.0
            importance = baseline_acc - acc_with_ablation
            layer_importance[layer_idx] = importance
            
            for h in hooks:
                h.remove()
        
        return {
            "baseline_accuracy": baseline_acc,
            "layer_importance": layer_importance,
            "critical_layers": sorted(
                layer_importance.items(),
                key=lambda x: x[1],
                reverse=True
            ),
        }
    
    def compare_activations(self, correct_question, incorrect_question, layer_idx):
        """
        Compare hidden states between correct and incorrect reasoning paths.
        
        Args:
            correct_question: Question where model answers correctly.
            incorrect_question: Question where model answers incorrectly.
            layer_idx: Which layer to analyze.
        
        Returns:
            Dictionary with activation statistics.
        """
        correct_acts = self._get_activations(correct_question, layer_idx)
        incorrect_acts = self._get_activations(incorrect_question, layer_idx)
        
        correct_acts = correct_acts.squeeze().cpu().numpy()
        incorrect_acts = incorrect_acts.squeeze().cpu().numpy()
        
        cosine_sim = np.dot(correct_acts, incorrect_acts) / (
            np.linalg.norm(correct_acts) * np.linalg.norm(incorrect_acts) + 1e-8
        )
        
        return {
            "correct_mean": float(correct_acts.mean()),
            "incorrect_mean": float(incorrect_acts.mean()),
            "correct_std": float(correct_acts.std()),
            "incorrect_std": float(incorrect_acts.std()),
            "cosine_similarity": float(cosine_sim),
            "l2_distance": float(np.linalg.norm(correct_acts - incorrect_acts)),
        }
    
    def _get_activations(self, question, layer_idx):
        """Get hidden state activations for a question at a specific layer."""
        inputs = self.tokenizer(question, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                output_hidden_states=True,
            )
        
        return outputs.hidden_states[layer_idx + 1]
    
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


class FailureAnalyzer:
    """Analyze patterns in reasoning failures."""
    
    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def categorize_failures(self, questions: List[str], answers: List[str]) -> Dict:
        """
        Categorize failures into types.
        
        Args:
            questions: List of question texts.
            answers: List of answer texts.
        
        Returns:
            Dictionary with failure categorization.
        """
        correct = []
        incorrect = []
        
        for q, a in zip(questions, answers):
            response = self._generate_response(q)
            if str(a).strip() in response:
                correct.append({"question": q, "answer": a, "response": response})
            else:
                incorrect.append({"question": q, "answer": a, "response": response})
        
        return {
            "total": len(questions),
            "correct_count": len(correct),
            "incorrect_count": len(incorrect),
            "accuracy": len(correct) / len(questions) if questions else 0.0,
            "correct_examples": correct,
            "incorrect_examples": incorrect,
        }
    
    def analyze_error_patterns(self, failures: List[Dict]) -> Dict:
        """
        Analyze error patterns from failed examples.
        
        Args:
            failures: List of failure dictionaries.
        
        Returns:
            Dictionary with error analysis.
        """
        patterns = {
            "off_by_one": 0,
            "wrong_operation": 0,
            "incomplete_reasoning": 0,
            "hallucination": 0,
        }
        
        for failure in failures:
            response = failure.get("response", "")
            expected = str(failure.get("answer", "")).strip()
            
            if response and expected:
                try:
                    response_num = int(response.split()[-1])
                    expected_num = int(expected)
                    
                    if abs(response_num - expected_num) == 1:
                        patterns["off_by_one"] += 1
                    else:
                        patterns["wrong_operation"] += 1
                except (ValueError, IndexError):
                    if len(response) < len(failure.get("question", "")):
                        patterns["incomplete_reasoning"] += 1
                    else:
                        patterns["hallucination"] += 1
        
        return {
            "pattern_counts": patterns,
            "total_failures": len(failures),
        }
    
    def _generate_response(self, question):
        """Generate model response."""
        inputs = self.tokenizer(question, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
