import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from transformers import DistilBertTokenizer, DistilBertModel
import torch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import os

# Load processed data
def load_processed_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Word2Vec Embeddings
def get_word2vec_embeddings(tokens, model):
    """Generate Word2Vec embeddings by averaging token vectors."""
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

def train_word2vec_model(tokens_list):
    """Train Word2Vec model on tokenized data."""
    model = Word2Vec(sentences=tokens_list, vector_size=100, window=5, min_count=1, workers=4)
    return model

# BERT Embeddings
def get_bert_embeddings(texts, tokenizer, model, device='cpu'):
    """Generate BERT embeddings using DistilBERT."""
    model.eval()
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())  # [CLS] token
    return np.vstack(embeddings)

# Train Logistic Regression with Word2Vec
def train_logistic_regression(X, y):
    """Train and evaluate Logistic Regression model."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    return model, {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

# Train LSTM with BERT
def train_lstm(X, y, input_dim=768):
    """Train and evaluate LSTM model."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = Sequential([
        LSTM(128, input_shape=(1, input_dim), return_sequences=False),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train.reshape(-1, 1, input_dim), y_train, epochs=5, batch_size=32, verbose=1)
    _, accuracy = model.evaluate(X_test.reshape(-1, 1, input_dim), y_test, verbose=0)
    y_pred = np.argmax(model.predict(X_test.reshape(-1, 1, input_dim)), axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    return model, {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

if __name__ == "__main__":
    # Load data
    df = load_processed_data("D:/sentiment_analysis_project/data/processed_data.csv")
    df = df.dropna(subset=['tokens', 'cleaned_text'])
    
    # Prepare labels
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    df['label'] = df['sentiment'].map(label_map)
    
    # Word2Vec: Train and get embeddings
    tokens_list = df['tokens'].apply(eval).tolist()  # Convert stringified list to list
    word2vec_model = train_word2vec_model(tokens_list)
    df['word2vec'] = df['tokens'].apply(lambda x: get_word2vec_embeddings(eval(x), word2vec_model))
    X_w2v = np.vstack(df['word2vec'].values)
    
    # BERT: Initialize tokenizer and model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
    X_bert = get_bert_embeddings(df['cleaned_text'].tolist(), tokenizer, bert_model, device)
    
    # Train and evaluate models
    y = df['label'].values
    lr_model, lr_metrics = train_logistic_regression(X_w2v, y)
    lstm_model, lstm_metrics = train_lstm(X_bert, y)
    
    # Save models
    os.makedirs("D:/sentiment_analysis_project/models", exist_ok=True)
    joblib.dump(lr_model, "D:/sentiment_analysis_project/models/logistic_regression_w2v.pkl")
    word2vec_model.save("D:/sentiment_analysis_project/models/word2vec.model")  # Save Word2Vec model
    lstm_model.save("D:/sentiment_analysis_project/models/lstm_bert.h5")  # Save LSTM model
    
    # Print metrics
    print("Logistic Regression (Word2Vec) Metrics:", lr_metrics)
    print("LSTM (BERT) Metrics:", lstm_metrics)