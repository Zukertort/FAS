import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

current_dir = os.path.dirname('test_model.ipynb')
resources_dir = os.path.join(current_dir, '..', 'resources')
csv_file_path = os.path.join(resources_dir, 'IMDB Dataset.csv')


try:
    df = pd.read_csv(csv_file_path)
    X = df['review']
    y = df['sentiment']
    print("Data loaded successfully.")

    print("Training the TfidfVectorizer...")
    vectorizer = TfidfVectorizer(max_features=5000)
    vectorizer.fit(X)
    print("Vectorizer training complete.")

    # Transform the text data
    X_vectorized = vectorizer.transform(X)

    # Train the model
    print("Training the MultinomialNB model...")
    model = MultinomialNB()
    model.fit(X_vectorized, y)
    print("Model training complete.")

    output_dir = 'nlp_model'
    os.makedirs(output_dir, exist_ok=True)

    # Save the vectorizer and the model
    joblib.dump(vectorizer, os.path.join(output_dir, 'vectorizer.joblib'))
    joblib.dump(model, os.path.join(output_dir, 'model.joblib'))

    print(f"\nModel and vectorizer have been saved to the '{output_dir}' directory.")
    print("--- Training Script Finished ---")

except FileNotFoundError:
    print(f"File '{csv_file_path}' not found.")

except Exception as e:
    print(f"An error occurred: {e}")

