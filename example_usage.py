import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

def example_with_pipeline():
    print("\n" + "="*60)
    print("METHOD 1: Using Pipeline (High-Level Helper)")
    print("="*60)
    
    pipe = pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis")
    
    test_texts = [
        "This is wonderful!",
        "I hate this product.",
        "यह बहुत अच्छा है",
        "C'est terrible",
        "Me encanta esto"
    ]
    
    for text in test_texts:
        result = pipe(text)
        print(f"\nText: {text}")
        print(f"Sentiment: {result[0]['label']} (Score: {result[0]['score']:.4f})")

def example_with_direct_model():
    print("\n" + "="*60)
    print("METHOD 2: Loading Model Directly (More Control)")
    print("="*60)
    
    tokenizer = AutoTokenizer.from_pretrained("tabularisai/multilingual-sentiment-analysis")
    model = AutoModelForSequenceClassification.from_pretrained("tabularisai/multilingual-sentiment-analysis")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    text = "This product exceeded my expectations!"
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = predictions[0][predicted_class].item()
    
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    
    print(f"\nText: {text}")
    print(f"Predicted Class: {predicted_class}")
    print(f"Sentiment: {label_map.get(predicted_class, 'unknown')}")
    print(f"Confidence: {confidence:.4f}")
    print(f"All probabilities: {predictions[0].cpu().numpy()}")

def example_batch_processing():
    print("\n" + "="*60)
    print("METHOD 3: Batch Processing")
    print("="*60)
    
    pipe = pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis")
    
    texts = [
        "I absolutely love this!",
        "This is okay, nothing special.",
        "Worst experience ever!",
        "Pretty good overall.",
        "Not bad, could be better."
    ]
    
    results = pipe(texts)
    
    print("\nBatch Processing Results:")
    for text, result in zip(texts, results):
        print(f"\nText: {text}")
        print(f"  → {result['label']} ({result['score']:.4f})")

if __name__ == "__main__":
    print("\n🌍 Multilingual Sentiment Analysis Demo")
    print("Model: tabularisai/multilingual-sentiment-analysis\n")
    
    example_with_pipeline()
    example_with_direct_model()
    example_batch_processing()
    
    print("\n" + "="*60)
    print("✅ All examples completed!")
    print("="*60)
