import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

print("Loading data...")
data = pd.read_excel('Datasets/dataset12.xlsx')
data = data.dropna(subset=['decision'])

print("Building TF-IDF model...")

def get_tfidf_features(text1, text2):
    tfidf = TfidfVectorizer(max_features=500)
    tfidf_matrix = tfidf.fit_transform([text1, text2])
    sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    combined = text1 + " " + text2
    return sim, combined

print("Computing features...")
resume_texts = data['Resume'].fillna('').astype(str).tolist()
jobdesc_texts = data['Job Description'].fillna('').astype(str).tolist()
transcript_texts = data['Transcript'].fillna('').astype(str).tolist()

all_texts = resume_texts + jobdesc_texts + transcript_texts
tfidf = TfidfVectorizer(max_features=300)
tfidf.fit(all_texts)

resume_tfidf = tfidf.transform(resume_texts)
jobdesc_tfidf = tfidf.transform(jobdesc_texts)
transcript_tfidf = tfidf.transform(transcript_texts)

features_list = []
for i in range(len(data)):
    resume_vec = resume_tfidf[i]
    jobdesc_vec = jobdesc_tfidf[i]
    transcript_vec = transcript_tfidf[i]
    
    resume_jobdesc_sim = cosine_similarity(resume_vec, jobdesc_vec)[0][0]
    transcript_jobdesc_sim = cosine_similarity(transcript_vec, jobdesc_vec)[0][0]
    resume_transcript_sim = cosine_similarity(resume_vec, transcript_vec)[0][0]
    
    resume_len = len(resume_texts[i])
    jobdesc_len = len(jobdesc_texts[i])
    transcript_len = len(transcript_texts[i])
    resume_words = len(resume_texts[i].split())
    transcript_words = len(transcript_texts[i].split())
    
    features_list.append([
        resume_len, jobdesc_len, transcript_len,
        resume_words, transcript_words,
        resume_jobdesc_sim, transcript_jobdesc_sim, resume_transcript_sim
    ])

X = np.array(features_list)
y = (data['decision'] == 'selected').astype(int).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training XGBoost with grid search...")
from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1]
}
grid = GridSearchCV(
    XGBClassifier(eval_metric='logloss', random_state=42),
    param_grid, cv=5, scoring='accuracy', n_jobs=-1
)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_

accuracy = best_model.score(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Best Params: {grid.best_params_}")

with open('model.pkl', 'wb') as f:
    pickle.dump({'model': best_model, 'tfidf': tfidf}, f)

print("Done! Model saved as model.pkl")
