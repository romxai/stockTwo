"""
Text embedding extraction using FinBERT
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import numpy as np
from typing import List, Tuple
from huggingface_hub import logout
import os


def clear_hf_cache():
    """Clear any cached HuggingFace credentials"""
    try:
        logout()
        print("✅ Logged out from HuggingFace")
    except:
        pass

    for key in ['HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HF_API_TOKEN']:
        if key in os.environ:
            del os.environ[key]


def load_finbert(model_name: str, device: torch.device):
    """
    Load FinBERT model for financial text analysis

    Args:
        model_name: Name of the FinBERT model
        device: Device to load model on

    Returns:
        model, tokenizer
    """
    print(f"🤖 Loading FinBERT model: {model_name}...")

    clear_hf_cache()

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=False)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, token=False)
        model.to(device)
        model.eval()

        print(f"✅ FinBERT loaded on {device}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

        return model, tokenizer

    except Exception as e:
        print(f"⚠️ Could not load FinBERT: {e}")
        print("   Trying alternative model (DistilBERT)...")

        model_name = 'distilbert-base-uncased'
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=False)
        model = AutoModel.from_pretrained(model_name, token=False)
        model.to(device)
        model.eval()

        print(f"✅ DistilBERT loaded on {device}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

        return model, tokenizer


def extract_finbert_features(texts: List[str],
                            model,
                            tokenizer,
                            device: torch.device,
                            batch_size: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract FinBERT embeddings and sentiment from news texts

    Args:
        texts: List of news text strings
        model: FinBERT model
        tokenizer: FinBERT tokenizer
        device: Device to run on
        batch_size: Batch size

    Returns:
        embeddings (N x 768), sentiments (N x 3)
    """
    print(f"🔍 Extracting FinBERT features from {len(texts)} texts...")

    embeddings_list = []
    sentiments_list = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]

        # Tokenize
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        # Extract features
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)

            # Get embeddings from [CLS] token
            embeddings = outputs.hidden_states[-1][:, 0, :].cpu().numpy()

            # Get sentiment probabilities
            sentiments = F.softmax(outputs.logits, dim=-1).cpu().numpy()

        embeddings_list.append(embeddings)
        sentiments_list.append(sentiments)

        if (i // batch_size + 1) % 10 == 0:
            print(f"   Processed {i + len(batch_texts)}/{len(texts)} texts...")

    all_embeddings = np.vstack(embeddings_list)
    all_sentiments = np.vstack(sentiments_list)

    print(f"✅ Extracted embeddings shape: {all_embeddings.shape}")
    print(f"   Sentiment shape: {all_sentiments.shape}")

    return all_embeddings, all_sentiments
