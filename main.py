import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Sample dataset for sentiment analysis
data = {
    'text': ["I love this product!", "It's terrible.", "Great experience.", "Not good."],
    'sentiment': ['positive', 'negative', 'positive', 'negative']
}

df = pd.DataFrame(data)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['sentiment'], test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train a Support Vector Machine (SVM) classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test_tfidf)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Example usage of the trained model
new_text = ["This is awesome!", "I'm not happy with it."]
new_text_tfidf = tfidf_vectorizer.transform(new_text)
predicted_sentiment = svm_classifier.predict(new_text_tfidf)
print(f'Predicted sentiment for new text: {predicted_sentiment}')
