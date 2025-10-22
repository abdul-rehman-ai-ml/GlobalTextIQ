from transformers import pipeline
import torch

def main():
    print("\n" + "="*70)
    print("🌍 MULTILINGUAL SENTIMENT ANALYSIS - INTERACTIVE DEMO")
    print("="*70)
    print("\nModel: tabularisai/multilingual-sentiment-analysis")
    
    device = 0 if torch.cuda.is_available() else -1
    device_name = "GPU (CUDA)" if device == 0 else "CPU"
    print(f"Device: {device_name}")
    
    print("\n⏳ Loading model... (this may take a moment on first run)")
    try:
        pipe = pipeline(
            "text-classification",
            model="tabularisai/multilingual-sentiment-analysis",
            device=device
        )
        print("✅ Model loaded successfully!\n")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\nPlease ensure you have installed dependencies:")
        print("  pip install -r requirements.txt")
        return
    
    print("="*70)
    print("Enter text to analyze sentiment (supports multiple languages)")
    print("Type 'quit', 'exit', or 'q' to stop")
    print("="*70)
    
    while True:
        print("\n" + "-"*70)
        user_input = input("\n📝 Enter text: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q', '']:
            print("\n👋 Thank you for using the sentiment analyzer!")
            print("="*70)
            break
        
        try:
            result = pipe(user_input)[0]
            sentiment = result['label']
            score = result['score']
            
            if 'positive' in sentiment.lower():
                emoji = "😊"
                color = "Positive"
            elif 'negative' in sentiment.lower():
                emoji = "😞"
                color = "Negative"
            else:
                emoji = "😐"
                color = "Neutral"
            
            print("\n" + "="*70)
            print(f"📊 ANALYSIS RESULTS")
            print("="*70)
            print(f"Text: {user_input}")
            print(f"\n{emoji} Sentiment: {color}")
            print(f"📈 Confidence: {score:.2%}")
            print(f"🏷️  Label: {sentiment}")
            print("="*70)
            
        except Exception as e:
            print(f"\n❌ Error analyzing text: {e}")
            print("Please try again with different text.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
