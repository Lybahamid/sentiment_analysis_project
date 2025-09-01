import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import joblib
from tensorflow.keras.models import load_model
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

# Set NLTK data path
nltk.data.path.append('C:/Users/Laiba H/AppData/Roaming/nltk_data')

# Download NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

def clean_text(text):
    """Clean text by removing URLs, mentions, hashtags, and punctuation."""
    if not isinstance(text, str):
        text = str(text) if pd.notna(text) else ''
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower().strip()
    return text if text else ''

def preprocess_text(text):
    """Tokenize, remove stopwords, and lemmatize text."""
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return tokens

def get_word2vec_embeddings(tokens, model):
    """Generate Word2Vec embeddings by averaging token vectors."""
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

def get_bert_embeddings(text, tokenizer, model, device='cpu'):
    """Generate BERT embeddings using DistilBERT."""
    model.eval()
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()  # [CLS] token

def predict_sentiment(texts, lr_model, word2vec_model, lstm_model, tokenizer, bert_model, device='cpu'):
    """Predict sentiment for a list of texts using both models."""
    label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    
    # Preprocess texts
    cleaned_texts = [clean_text(text) for text in texts]
    tokenized_texts = [preprocess_text(text) for text in cleaned_texts]
    
    # Logistic Regression predictions (Word2Vec)
    lr_embeddings = np.vstack([get_word2vec_embeddings(tokens, word2vec_model) for tokens in tokenized_texts])
    lr_predictions = lr_model.predict(lr_embeddings)
    lr_labels = [label_map[pred] for pred in lr_predictions]
    
    # LSTM predictions (BERT)
    bert_embeddings = get_bert_embeddings(cleaned_texts, tokenizer, bert_model, device)
    lstm_predictions = np.argmax(lstm_model.predict(bert_embeddings.reshape(-1, 1, 768)), axis=1)
    lstm_labels = [label_map[pred] for pred in lstm_predictions]
    
    return list(zip(texts, lr_labels, lstm_labels))

if __name__ == "__main__":
    # Load models
    lr_model = joblib.load("D:/sentiment_analysis_project/models/logistic_regression_w2v.pkl")
    lstm_model = load_model("D:/sentiment_analysis_project/models/lstm_bert.h5")
    word2vec_model = Word2Vec.load("D:/sentiment_analysis_project/models/word2vec.model")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bert_model.to(device)
    
    # Save Word2Vec model (needed for inference, not saved in train.py)
    word2vec_model = Word2Vec.load("D:/sentiment_analysis_project/models/word2vec.model")
    
    # Sample texts for inference
    sample_texts = [
        "I love this product, it's amazing!",
        "This is the worst experience ever.",
        "The weather is okay today."
    ]
    
    # Predict sentiments
    predictions = predict_sentiment(sample_texts, lr_model, word2vec_model, lstm_model, tokenizer, bert_model, device)
    
    # Print results
    print("\nSentiment Predictions:")
    print("Text | Logistic Regression | LSTM (BERT)")
    print("-" * 50)
    for text, lr_pred, lstm_pred in predictions:
        print(f"{text[:50]:50} | {lr_pred:20} | {lstm_pred}")