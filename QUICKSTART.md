# 🚀 Quick Start Guide

## Step 1: Install Dependencies

Open your terminal in this directory and run:

```bash
pip install -r requirements.txt
```

**Note**: First time installation will download the model (~500MB)

---

## Step 2: Choose Your Method

### 🎮 Option A: Interactive Demo (Easiest)

```bash
python interactive_demo.py
```

Type any text in any language and see the sentiment analysis results instantly!

**Example**:
```
Enter text: I love this product!
Result: 😊 Positive (98.7% confidence)
```

---

### 🧪 Option B: Run Tests

```bash
python test_model.py
```

This will test the model with pre-defined examples in multiple languages.

---

### 📚 Option C: See All Examples

```bash
python example_usage.py
```

This demonstrates 3 different ways to use the model:
1. Pipeline API
2. Direct model loading
3. Batch processing

---

### 🔧 Option D: Run Main Pipeline

```bash
python nlp_pipeline.py
```

This runs the main pipeline with multilingual test cases.

---

## Step 3: Use in Your Code

### Simplest Way:

```python
from transformers import pipeline

# Load the model
pipe = pipeline("text-classification", 
                model="tabularisai/multilingual-sentiment-analysis")

# Analyze sentiment
result = pipe("I love this!")
print(result)
# Output: [{'label': 'positive', 'score': 0.9987}]
```

### Using the Custom Class:

```python
from nlp_pipeline import NLPPipeline

# Initialize
nlp = NLPPipeline()
nlp.load_pipeline(
    model_name="tabularisai/multilingual-sentiment-analysis",
    task="text-classification"
)

# Predict
result = nlp.predict("This is amazing!")
print(result)
```

---

## 🌍 Supported Languages

The model works with multiple languages:
- ✅ English
- ✅ Hindi (हिंदी)
- ✅ French (Français)
- ✅ Spanish (Español)
- ✅ German (Deutsch)
- ✅ And many more!

---

## 💡 Tips

1. **GPU Support**: The model will automatically use GPU if available (much faster!)
2. **Batch Processing**: Process multiple texts at once for better performance
3. **First Run**: Model download happens on first run (be patient!)
4. **Confidence Scores**: Higher scores mean more confident predictions

---

## 🆘 Troubleshooting

### Issue: Module not found
**Solution**: Make sure you installed dependencies:
```bash
pip install -r requirements.txt
```

### Issue: Out of memory
**Solution**: Use CPU instead of GPU or reduce batch size in `config.py`

### Issue: Model download fails
**Solution**: Check your internet connection and try again

---

## 📖 Need More Help?

- Check `README.md` for detailed documentation
- See `PROJECT_SUMMARY.md` for project overview
- Look at `example_usage.py` for code examples

---

## ✨ You're Ready!

Start with the interactive demo:
```bash
python interactive_demo.py
```

Happy analyzing! 🎉
