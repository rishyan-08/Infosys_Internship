import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Recruitment Screener", page_icon=":", layout="wide")

st.title("AI-Driven Recruitment Pipeline")
st.markdown("Predict whether a candidate should be **Selected** or **Rejected** based on their resume and job description.")

@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['tfidf']

def extract_features(resume, job_description, tfidf):
    resume_vec = tfidf.transform([resume])
    jobdesc_vec = tfidf.transform([job_description])
    
    resume_jobdesc_sim = cosine_similarity(resume_vec, jobdesc_vec)[0][0]
    
    resume_len = len(resume)
    jobdesc_len = len(job_description)
    resume_words = len(resume.split())
    
    transcript_len = 0
    transcript_words = 0
    transcript_jobdesc_sim = 0
    resume_transcript_sim = 0
    
    features = np.array([[
        resume_len, jobdesc_len, transcript_len,
        resume_words, transcript_words,
        resume_jobdesc_sim, transcript_jobdesc_sim, resume_transcript_sim
    ]])
    return features, resume_jobdesc_sim

model, tfidf = load_model()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Candidate Resume")
    resume = st.text_area("Paste the candidate's resume here:", height=300, key="resume")

with col2:
    st.subheader("Job Description")
    job_description = st.text_area("Paste the job description here:", height=300, key="jobdesc")

if st.button("Predict", type="primary"):
    if resume.strip() == "" or job_description.strip() == "":
        st.warning("Please fill in both fields.")
    else:
        features, sim_score = extract_features(resume, job_description, tfidf)
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]

        st.markdown("---")
        st.subheader("Result")

        if prediction == 1:
            st.success("Selected")
        else:
            st.error("Rejected")

        st.metric("Confidence", f"{max(probability)*100:.1f}%")

        st.markdown("---")
        st.subheader("Feature Analysis")
        st.write(f"- Resume Length: {len(resume)} characters")
        st.write(f"- Job Description Length: {len(job_description)} characters")
        st.write(f"- Resume-Job Similarity: {sim_score:.3f}")

st.markdown("---")
st.markdown("Built with TF-IDF, XGBoost, and Streamlit | Accuracy: 81.6%")
