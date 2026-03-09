import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("news_dataset.csv")

# Features and labels
X = data["text"]
y = data["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Convert text to numbers using TF-IDF
vectorizer = TfidfVectorizer(stop_words="english",max_features=5000)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Test model
predictions = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)

# Function to predict news
def predict_news(news):
    news_vector = vectorizer.transform([news])
    prediction = model.predict(news_vector)

    if prediction[0] == 1:
        print("✅ This looks like REAL news")
    else:
        print("⚠️ This looks like FAKE news")


# User input
user_news = input("\nEnter a news headline:humans claim 5G technology was linked to spread of covid-19")
predict_news(user_news)
