import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

class NLPPipeline:
    def __init__(self, model_name: str = None, task: str = None, device: int = -1):
        self.model_name = model_name
        self.task = task
        self.device = device
        self.pipeline = None
        self.tokenizer = None
        self.model = None
        
    def load_pipeline(self, model_name: str = None, task: str = None):
        if model_name:
            self.model_name = model_name
        if task:
            self.task = task
            
        if not all([self.model_name, self.task]):
            raise ValueError("Both model_name and task must be provided")
            
        self.pipeline = pipeline(
            task=self.task,
            model=self.model_name,
            device=self.device
        )
        return self.pipeline
    
    def predict(self, text: str, **kwargs):
        if not self.pipeline:
            raise ValueError("Pipeline not loaded. Call load_pipeline() first.")
        return self.pipeline(text, **kwargs)
    
    def load_custom_model(self, model_name: str = None, tokenizer_name: str = None):
        self.model_name = model_name or self.model_name
        if not self.model_name:
            raise ValueError("model_name must be provided")
            
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        
        if self.device >= 0 and torch.cuda.is_available():
            self.model = self.model.to(f"cuda:{self.device}")
            
        return self.model, self.tokenizer


def main():
    nlp = NLPPipeline(device=0 if torch.cuda.is_available() else -1)
    
    model_name = "tabularisai/multilingual-sentiment-analysis"
    nlp.load_pipeline(model_name=model_name, task="text-classification")
    
    test_texts = [
        "This is a positive sentence!",
        "I love this product, it's amazing!",
        "This is terrible and disappointing.",
        "यह बहुत अच्छा है",
        "C'est fantastique!"
    ]
    
    print("=" * 60)
    print("Multilingual Sentiment Analysis Results")
    print("=" * 60)
    
    for text in test_texts:
        result = nlp.predict(text)
        print(f"\nText: {text}")
        print(f"Result: {result}")
    
    print("\n" + "=" * 60)
    print("Pipeline ready for use!")

if __name__ == "__main__":
    main()
