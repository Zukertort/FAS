# test_vader.py

# Import the necessary library
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def analyze_sentiment(sentence_list):
    """
    Analyzes the sentiment of a list of sentences using VADER.
    """
    # Initialize the VADER Sentiment Analyzer
    analyzer = SentimentIntensityAnalyzer()
    
    print("--- VADER Sentiment Analysis Results ---")
    print("-" * 40)
    
    # Loop through each sentence and analyze
    for i, sentence in enumerate(sentence_list):
        # The polarity_scores() method returns a dictionary of scores.
        sentiment_scores = analyzer.polarity_scores(sentence)
        
        # The 'compound' score is a single, normalized metric.
        # It ranges from -1 (most extreme negative) to +1 (most extreme positive).
        compound_score = sentiment_scores['compound']
        
        # Assign a sentiment label based on the compound score
        if compound_score >= 0.05:
            sentiment_label = "Positive"
        elif compound_score <= -0.05:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"
            
        # Print the results for each sentence
        print(f"Sentence {i+1}: \"{sentence}\"")
        print(f"   -> Compound Score: {compound_score}")
        print(f"   -> Final Label: {sentiment_label}")
        print("-" * 40)

# --- Main execution block ---
if __name__ == "__main__":
    # Define a sample dataset of sentences
    sample_data = [
        "The movie was absolutely fantastic! I loved every minute of it.",
        "Customer service was abysmal. I am never returning to that store.",
        "The weather today is just okay, neither sunny nor rainy.",
        "I'm feeling incredibly excited about the upcoming vacation!",
        "He was late for the meeting and didn't seem prepared at all.",
        "This new update is fantastic! It fixed all the major bugs. 😊",
        "The book was just announced for release next Tuesday.",
        "What a complete and utter waste of my time and money.",
        "I'm not sure how I feel about the new company policy."
    ]
    
    # Run the analysis on our sample data
    analyze_sentiment(sample_data)