import numpy as np
import pandas as pd
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

print("Loading data...")
data = pd.read_excel('dataset1.xlsx')

print("Preprocessing...")
data['decision'] = data['decision'].replace({'reject': 'rejected', 'select': 'selected'})
data = data.dropna(subset=['Transcript', 'Resume', 'Job Description', 'decision'])

def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'[^\w\s]', '', text).lower()
        return text
    return ""

for col in ['Transcript', 'Resume', 'Job Description']:
    data[col] = data[col].apply(clean_text)

print("Extracting features...")
data['length_of_transcript'] = data['Transcript'].apply(len)
data['num_words_in_transcript'] = data['Transcript'].apply(lambda x: len(x.split()))
data['length_of_resume'] = data['Resume'].apply(len)
data['num_words_in_resume'] = data['Resume'].apply(lambda x: len(x.split()))

def calculate_similarity(text1, text2):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

print("Computing similarities (this may take a while)...")
data['resume_transcript_similarity'] = data.apply(lambda x: calculate_similarity(x['Resume'], x['Transcript']), axis=1)
data['transcript_jobdesc_similarity'] = data.apply(lambda x: calculate_similarity(x['Transcript'], x['Job Description']), axis=1)
data['resume_jobdesc_similarity'] = data.apply(lambda x: calculate_similarity(x['Resume'], x['Job Description']), axis=1)

feature_cols = ['num_words_in_transcript', 'length_of_transcript', 'num_words_in_resume',
                'length_of_resume', 'resume_transcript_similarity',
                'transcript_jobdesc_similarity', 'resume_jobdesc_similarity']

X = data[feature_cols]
le = LabelEncoder()
y = le.fit_transform(data['decision'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training XGBoost...")
model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

print("Saving model...")
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Done! Model saved as model.pkl")
