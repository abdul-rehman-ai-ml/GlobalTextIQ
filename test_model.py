from transformers import pipeline
import torch

def test_model():
    print("🚀 Testing Multilingual Sentiment Analysis Model")
    print("=" * 60)
    
    device = 0 if torch.cuda.is_available() else -1
    device_name = "GPU (CUDA)" if device == 0 else "CPU"
    print(f"Using device: {device_name}\n")
    
    print("Loading model: tabularisai/multilingual-sentiment-analysis...")
    pipe = pipeline(
        "text-classification",
        model="tabularisai/multilingual-sentiment-analysis",
        device=device
    )
    print("✅ Model loaded successfully!\n")
    
    test_cases = [
        ("English", "I absolutely love this product!"),
        ("English", "This is the worst thing I've ever bought."),
        ("English", "It's okay, nothing special."),
        ("Hindi", "यह बहुत अच्छा है"),
        ("French", "C'est fantastique!"),
        ("Spanish", "Me encanta esto"),
        ("German", "Das ist schrecklich"),
    ]
    
    print("Running sentiment analysis tests:")
    print("-" * 60)
    
    for language, text in test_cases:
        result = pipe(text)[0]
        sentiment = result['label']
        score = result['score']
        
        emoji = "😊" if "positive" in sentiment.lower() else "😞" if "negative" in sentiment.lower() else "😐"
        
        print(f"\n[{language}] {text}")
        print(f"  {emoji} Sentiment: {sentiment} | Confidence: {score:.2%}")
    
    print("\n" + "=" * 60)
    print("✅ All tests completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        test_model()
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        print("\nMake sure you have installed all dependencies:")
        print("  pip install -r requirements.txt")
