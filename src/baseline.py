import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def generate_response(model, tokenizer, prompt, max_tokens=50, device="cpu"):
    """
    Generate a response from the model.
    
    Args:
        model: The language model.
        tokenizer: The tokenizer.
        prompt: Input text.
        max_tokens: Maximum tokens to generate.
        device: Device to use (cpu or cuda).
    
    Returns:
        Generated text.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def collect_activations(model, tokenizer, prompt, device="cpu"):
    """
    Collect hidden activations and attention weights during forward pass.
    
    Args:
        model: The language model.
        tokenizer: The tokenizer.
        prompt: Input text.
        device: Device to use.
    
    Returns:
        Dictionary with activations, attention weights, and token ids.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            output_hidden_states=True,
            output_attentions=True,
        )
    
    result = {
        "token_ids": inputs.input_ids,
        "tokens": tokenizer.convert_ids_to_tokens(inputs.input_ids[0]),
        "hidden_states": [hs.detach().cpu() for hs in outputs.hidden_states],
        "attention_weights": [attn.detach().cpu() for attn in outputs.attentions],
        "logits": outputs.logits.detach().cpu(),
    }
    
    return result


def get_model_accuracy(model, tokenizer, questions, answers, device="cpu", max_tokens=50):
    """
    Measure model accuracy on a set of questions.
    
    Args:
        model: The language model.
        tokenizer: The tokenizer.
        questions: List of question strings.
        answers: List of correct answer strings.
        device: Device to use.
        max_tokens: Max tokens to generate.
    
    Returns:
        Accuracy as a float between 0 and 1.
    """
    correct = 0
    
    for question, answer in zip(questions, answers):
        response = generate_response(model, tokenizer, question, max_tokens, device)
        if str(answer).strip() in response:
            correct += 1
    
    return correct / len(questions) if questions else 0.0
