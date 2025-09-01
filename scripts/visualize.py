import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os

def plot_sentiment_distribution(df):
    """Plot sentiment distribution."""
    try:
        plt.figure(figsize=(8, 6))
        sns.countplot(x='sentiment', data=df, order=['positive', 'neutral', 'negative'])
        plt.title('Sentiment Distribution')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.savefig('D:/sentiment_analysis_project/outputs/sentiment_distribution.png')
        plt.close()
    except Exception as e:
        print(f"Error in plot_sentiment_distribution: {e}")

def generate_wordclouds(df):
    """Generate word clouds for positive and negative sentiments."""
    try:
        # Filter and convert to string, excluding NaN values
        positive_text = ' '.join(df[df['sentiment'] == 'positive']['cleaned_text'].dropna().astype(str))
        negative_text = ' '.join(df[df['sentiment'] == 'negative']['cleaned_text'].dropna().astype(str))
        
        # Check if text is empty
        if not positive_text.strip():
            print("Warning: No valid text for positive sentiment word cloud")
            return
        if not negative_text.strip():
            print("Warning: No valid text for negative sentiment word cloud")
            return
        
        # Positive word cloud
        wordcloud_pos = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_pos, interpolation='bilinear')
        plt.axis('off')
        plt.title('Positive Sentiment Word Cloud')
        plt.savefig('D:/sentiment_analysis_project/outputs/wordcloud_positive.png')
        plt.close()
        
        # Negative word cloud
        wordcloud_neg = WordCloud(width=800, height=400, background_color='white').generate(negative_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_neg, interpolation='bilinear')
        plt.axis('off')
        plt.title('Negative Sentiment Word Cloud')
        plt.savefig('D:/sentiment_analysis_project/outputs/wordcloud_negative.png')
        plt.close()
    except Exception as e:
        print(f"Error in generate_wordclouds: {e}")

if __name__ == "__main__":
    # Load processed data
    try:
        df = pd.read_csv("D:/sentiment_analysis_project/data/processed_data.csv")
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)
    
    # Create output directory
    os.makedirs("D:/sentiment_analysis_project/outputs", exist_ok=True)
    
    # Generate visualizations
    plot_sentiment_distribution(df)
    generate_wordclouds(df)
    print("Visualizations saved in D:/sentiment_analysis_project/outputs/")