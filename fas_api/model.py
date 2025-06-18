# import os
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# import joblib

# current_dir = os.path.dirname('test_model.ipynb')
# resources_dir = os.path.join(current_dir, '..', 'resources')
# csv_file_path = os.path.join(resources_dir, 'IMDB Dataset.csv')


# try:
#     df = pd.read_csv(csv_file_path)
#     X = df['review']
#     y = df['sentiment']
#     print("Data loaded successfully.")

#     print("Training the TfidfVectorizer...")
#     vectorizer = TfidfVectorizer(max_features=5000)
#     vectorizer.fit(X)
#     print("Vectorizer training complete.")

#     # Transform the text data
#     X_vectorized = vectorizer.transform(X)

#     # Train the model
#     print("Training the MultinomialNB model...")
#     model = MultinomialNB()
#     model.fit(X_vectorized, y)
#     print("Model training complete.")

#     output_dir = 'nlp_model'
#     os.makedirs(output_dir, exist_ok=True)

#     # Save the vectorizer and the model
#     joblib.dump(vectorizer, os.path.join(output_dir, 'vectorizer.joblib'))
#     joblib.dump(model, os.path.join(output_dir, 'model.joblib'))

#     print(f"\nModel and vectorizer have been saved to the '{output_dir}' directory.")
#     print("--- Training Script Finished ---")

# except FileNotFoundError:
#     print(f"File '{csv_file_path}' not found.")

# except Exception as e:
#     print(f"An error occurred: {e}")

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

current_dir = os.path.dirname('test_model.ipynb')
resources_dir = os.path.join(current_dir, '..', 'resources')
csv_file_path = os.path.join(resources_dir, 'IMDB Dataset.csv')

df = pd.read_csv(csv_file_path)

X = df['review']
y = df['sentiment']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
classification = classification_report(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
cm_normalized = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]

print("\nAccuracy:", accuracy)
print("\nClassification Report:\n", classification)

# Create a figure and a set of subplots: 1 row, 2 columns
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Raw Confusion Matrix (on the left, axes[0])
sns.heatmap(confusion, annot=True, cmap='Blues', fmt='d', ax=axes[0]) # fmt='d' ensures integer formatting
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('Actual Label')
axes[0].set_title('Confusion Matrix (Counts)')
axes[0].set_xticklabels(['Predicted Negative', 'Predicted Positive'])
axes[0].set_yticklabels(['Actual Negative', 'Actual Positive'])

# Plot 2: Normalized Confusion Matrix (on the right, axes[1])
sns.heatmap(
    cm_normalized, annot=True, fmt='.2%', cmap='Blues', cbar=True,
    xticklabels=['Predicted Negative', 'Predicted Positive'],
    yticklabels=['Actual Negative', 'Actual Positive'],
    ax=axes[1] # Crucially, specify which subplot to draw on
)
axes[1].set_xlabel('Predicted Label')
axes[1].set_ylabel('Actual Label')
axes[1].set_title('Normalized Confusion Matrix (%)')

# Adjust layout to prevent titles/labels from overlapping
plt.tight_layout()

# Show the plots
plt.show()
