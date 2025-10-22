import torch
from typing import List, Dict, Any, Union
import numpy as np
from tqdm import tqdm

def batch_texts(texts: List[str], batch_size: int = 8) -> List[List[str]]:
    return [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

def get_device(use_gpu: bool = True) -> torch.device:
    if use_gpu and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def process_batch(
    model,
    tokenizer,
    texts: List[str],
    max_length: int = 512,
    truncation: bool = True,
    padding: bool = True,
    return_tensors: str = 'pt'
) -> Dict[str, torch.Tensor]:
    return tokenizer(
        texts,
        max_length=max_length,
        truncation=truncation,
        padding=padding,
        return_tensors=return_tensors
    )

def predict_batch(
    model,
    tokenizer,
    texts: List[str],
    batch_size: int = 8,
    show_progress: bool = True,
    **kwargs
) -> List[Dict[str, Any]]:
    model.eval()
    device = next(model.parameters()).device
    results = []
    
    batches = batch_texts(texts, batch_size)
    if show_progress:
        batches = tqdm(batches, desc="Processing batches")
    
    with torch.no_grad():
        for batch in batches:
            inputs = process_batch(model, tokenizer, batch, **kwargs)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            logits = outputs.logits
            predictions = torch.softmax(logits, dim=-1)
            
            for i in range(len(batch)):
                results.append({
                    'text': batch[i],
                    'probabilities': predictions[i].cpu().numpy(),
                    'prediction': torch.argmax(predictions[i]).item()
                })
    
    return results

def print_prediction_results(results: List[Dict[str, Any]], label_map: dict = None) -> None:
    for i, result in enumerate(results):
        print(f"\nText {i+1}:")
        print(f"  Input: {result['text']}")
        
        if label_map:
            pred_label = label_map.get(result['prediction'], f"Label {result['prediction']}")
            print(f"  Predicted: {pred_label} (Confidence: {np.max(result['probabilities']):.4f})")
        else:
            print(f"  Prediction: {result['prediction']} (Confidence: {np.max(result['probabilities']):.4f})")
        
        if 'probabilities' in result and label_map:
            print("  Probabilities:")
            for idx, prob in enumerate(result['probabilities']):
                label = label_map.get(idx, f"Label {idx}")
                print(f"    {label}: {prob:.4f}")
