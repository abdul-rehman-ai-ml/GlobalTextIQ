MODEL_CONFIG = {
    "model_name": "tabularisai/multilingual-sentiment-analysis",
    "task": "text-classification",
    "max_length": 512,
    "truncation": True,
    "padding": True,
}

DEVICE_CONFIG = {
    "use_gpu": True,
    "device_id": 0,
}

INFERENCE_CONFIG = {
    "batch_size": 8,
    "show_progress": True,
}

LABEL_MAP = {
    0: "negative",
    1: "neutral",
    2: "positive",
}

SAMPLE_TEXTS = {
    "english": [
        "This is amazing!",
        "I hate this.",
        "It's okay.",
    ],
    "hindi": [
        "यह बहुत अच्छा है",
        "यह बुरा है",
    ],
    "french": [
        "C'est fantastique!",
        "C'est terrible",
    ],
    "spanish": [
        "Me encanta esto",
        "Esto es horrible",
    ],
}
