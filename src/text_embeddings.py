"""
Text embedding extraction using FinBERT

FinBERT is a BERT model fine-tuned on financial text (10-K reports, earnings calls,
analyst reports) that understands financial sentiment and language better than
general-purpose BERT models. It outputs:
1. Dense embeddings (768-dim vectors) capturing semantic meaning
2. Sentiment probabilities (positive, negative, neutral)

These features help the model understand market sentiment from news.
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import numpy as np
from typing import List, Tuple
from huggingface_hub import logout
import os


def clear_hf_cache():
    """
    Clear any cached HuggingFace credentials
    
    Removes authentication tokens to avoid permission issues when loading
    public models. Some models may require authentication, but FinBERT is public.
    """
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
    
    Attempts to load FinBERT ('yiyanghkust/finbert-tone') which classifies
    financial text into positive/negative/neutral sentiment. Falls back to
    DistilBERT if FinBERT is unavailable.
    
    Model is set to eval mode for inference (disables dropout, batch norm updates).

    Args:
        model_name: HuggingFace model identifier (e.g., 'yiyanghkust/finbert-tone')
        device: torch.device to load model on (cuda or cpu)

    Returns:
        Tuple of (model, tokenizer)
        - model: Pre-trained transformer with classification head
        - tokenizer: Converts text to token IDs
        
    Raises:
        Exception: If model loading fails (caught and falls back to DistilBERT)
    """
    print(f"🤖 Loading FinBERT model: {model_name}...")

    # Remove any cached authentication tokens
    clear_hf_cache()

    try:
        # Load tokenizer (converts text to token IDs)
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=False)
        # Load model with classification head (for sentiment)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, token=False)
        model.to(device)
        model.eval()  # Set to evaluation mode (disables dropout)

        print(f"✅ FinBERT loaded on {device}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

        return model, tokenizer

    except Exception as e:
        # Fallback to DistilBERT if FinBERT is unavailable
        print(f"⚠️ Could not load FinBERT: {e}")
        print("   Trying alternative model (DistilBERT)...")

        model_name = 'distilbert-base-uncased'
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=False)
        # Note: Using base model (no classification head) for fallback
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
    
    Processes texts in batches to extract:
    1. Embeddings: 768-dimensional vectors from [CLS] token (semantic representation)
    2. Sentiments: Probability distribution over [positive, negative, neutral]
    
    The [CLS] token is special: BERT prepends it to every sequence, and its
    embedding captures the overall meaning of the entire text.
    
    Uses gradient-free inference (torch.no_grad) for speed and memory efficiency.

    Args:
        texts: List of news article strings (can be empty strings for missing news)
        model: Pre-trained FinBERT model
        tokenizer: FinBERT tokenizer for text preprocessing
        device: torch.device (cuda or cpu)
        batch_size: Number of texts to process at once (lower for GPU memory limits)

    Returns:
        Tuple of:
        - embeddings: numpy array (N, 768) - semantic embeddings
        - sentiments: numpy array (N, 3) - [positive_prob, negative_prob, neutral_prob]
    """
    print(f"🔍 Extracting FinBERT features from {len(texts)} texts...")

    embeddings_list = []
    sentiments_list = []

    # Process texts in batches to manage memory
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]

        # Tokenize: convert text to token IDs
        # padding=True: pad shorter sequences to match longest in batch
        # truncation=True: cut sequences longer than max_length
        # max_length=512: BERT's maximum context length
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'  # Return PyTorch tensors
        )

        # Move inputs to GPU/CPU
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)  # Tells model which tokens are padding

        # Extract features without computing gradients (inference only)
        with torch.no_grad():
            # output_hidden_states=True: return all layer outputs, not just final
            outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)

            # Extract [CLS] token embedding from last layer
            # hidden_states[-1] = last transformer layer output
            # [:, 0, :] = first token ([CLS]) for all samples in batch
            embeddings = outputs.hidden_states[-1][:, 0, :].cpu().numpy()

            # Get sentiment probabilities from classification head
            # Softmax converts logits to probabilities that sum to 1
            sentiments = F.softmax(outputs.logits, dim=-1).cpu().numpy()

        embeddings_list.append(embeddings)
        sentiments_list.append(sentiments)

        # Progress logging every 10 batches
        if (i // batch_size + 1) % 10 == 0:
            print(f"   Processed {i + len(batch_texts)}/{len(texts)} texts...")

    all_embeddings = np.vstack(embeddings_list)
    all_sentiments = np.vstack(sentiments_list)

    print(f"✅ Extracted embeddings shape: {all_embeddings.shape}")
    print(f"   Sentiment shape: {all_sentiments.shape}")

    return all_embeddings, all_sentiments
