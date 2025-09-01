import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os

# Set NLTK data path
nltk.data.path.append('C:/Users/Laiba H/AppData/Roaming/nltk_data')

# Download NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(file_path, sample_size=10000):
    """Load and sample the dataset."""
    try:
        df = pd.read_csv(file_path, encoding='latin-1', names=['sentiment', 'id', 'date', 'flag', 'user', 'text'], on_bad_lines='skip')
        df = df[['sentiment', 'text', 'date']].sample(n=sample_size, random_state=42)
        df['sentiment'] = df['sentiment'].map({0: 'negative', 2: 'neutral', 4: 'positive'})
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def clean_text(text):
    """Clean text by removing URLs, mentions, hashtags, and punctuation."""
    if not isinstance(text, str):
        text = str(text) if pd.notna(text) else ''
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower().strip()
    return text if text else ''  # Return empty string if text is empty

def preprocess_text(text):
    """Tokenize, remove stopwords, and lemmatize text."""
    try:
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
        return tokens
    except Exception as e:
        print(f"Error in preprocessing text: {e}")
        return []

def preprocess_data(df):
    """Apply cleaning and preprocessing to the dataset."""
    df['cleaned_text'] = df['text'].apply(clean_text)
    df['tokens'] = df['cleaned_text'].apply(preprocess_text)
    return df

if __name__ == "__main__":
    # Define paths
    data_path = "D:/sentiment_analysis_project/data/training.1600000.processed.noemoticon.csv"
    output_path = "D:/sentiment_analysis_project/data/processed_data.csv"
    
    # Load and preprocess data
    df = load_data(data_path)
    df = preprocess_data(df)
    
    # Save processed data
    os.makedirs("D:/sentiment_analysis_project/data", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")