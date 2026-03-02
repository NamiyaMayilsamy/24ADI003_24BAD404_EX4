import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

df = pd.read_csv(r"C:\Users\namiy\Downloads\spam.csv", encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['cleaned_message'] = df['message'].apply(clean_text)

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['cleaned_message'])

encoder = LabelEncoder()
y = encoder.fit_transform(df['label'])

X_train, X_test, y_train, y_test, train_index, test_index = train_test_split(
    X, y, df.index, test_size=0.2, random_state=42
)

model = MultinomialNB(alpha=1.0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

misclassified = df.iloc[test_index][y_test != y_pred]
print("\nNumber of Misclassified Messages:", len(misclassified))
print(misclassified[['label', 'message']].head(10))

alphas = [0.01, 0.1, 1, 5, 10]
print("\nLaplace Smoothing Impact:")
for a in alphas:
    temp_model = MultinomialNB(alpha=a)
    temp_model.fit(X_train, y_train)
    temp_pred = temp_model.predict(X_test)
    print("Alpha =", a, "Accuracy =", accuracy_score(y_test, temp_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

feature_names = vectorizer.get_feature_names_out()
spam_prob = model.feature_log_prob_[1]
top_indices = np.argsort(spam_prob)[-15:]
top_words = feature_names[top_indices]
top_scores = spam_prob[top_indices]

sorted_idx = np.argsort(top_scores)
plt.figure()
plt.barh(top_words[sorted_idx], top_scores[sorted_idx])
plt.xlabel("Log Probability")
plt.title("Top Words Influencing Spam Classification")
plt.show()

spam_messages = df[df['label'] == 'spam']['cleaned_message']
ham_messages = df[df['label'] == 'ham']['cleaned_message']

spam_vectorizer = CountVectorizer(stop_words='english', max_features=15)
ham_vectorizer = CountVectorizer(stop_words='english', max_features=15)

spam_counts = spam_vectorizer.fit_transform(spam_messages)
ham_counts = ham_vectorizer.fit_transform(ham_messages)

spam_sum = np.array(spam_counts.sum(axis=0)).flatten()
ham_sum = np.array(ham_counts.sum(axis=0)).flatten()

plt.figure()
plt.bar(spam_vectorizer.get_feature_names_out(), spam_sum)
plt.xticks(rotation=45)
plt.title("Top Words in Spam Messages")
plt.show()

plt.figure()
plt.bar(ham_vectorizer.get_feature_names_out(), ham_sum)
plt.xticks(rotation=45)
plt.title("Top Words in Ham Messages")
plt.show()
