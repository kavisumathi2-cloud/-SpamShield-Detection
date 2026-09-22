import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Sample Spam Data
data = {
    'message': [
        'Free entry in 2 a weekly competition to win FA Cup tickets',
        'Hey, are we still meeting for lunch today?',
        'Congratulations, you have won a free iPhone! Click here to claim.',
        'Can you send me the project notes by tonight?',
        'URGENT! Your mobile account has won £1000 cash. Reply WIN',
        'Let me know when you reach home safely, mom.'
    ],
    'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham']
}

df = pd.DataFrame(data)

# Text Vectorization
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['label']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Model Training
model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Model Training Completed Successfully! ✅")
print("Model Accuracy Score:", accuracy_score(y_test, y_pred))
