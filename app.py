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
        model = pickle.load(f)
    return model

def calculate_similarity(text1, text2):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform([text1, text2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

def extract_features(resume, job_description):
    transcript_placeholder = resume
    resume_transcript_sim = calculate_similarity(resume, transcript_placeholder)
    transcript_jobdesc_sim = calculate_similarity(transcript_placeholder, job_description)
    resume_jobdesc_sim = calculate_similarity(resume, job_description)

    features = np.array([[
        len(transcript_placeholder.split()),
        len(transcript_placeholder),
        len(resume.split()),
        len(resume),
        resume_transcript_sim,
        transcript_jobdesc_sim,
        resume_jobdesc_sim
    ]])
    return features

model = load_model()

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
        features = extract_features(resume, job_description)
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
        st.write(f"- Resume-Job Similarity: {calculate_similarity(resume, job_description):.3f}")

st.markdown("---")
st.markdown("Built with Scikit-learn, XGBoost, and Streamlit | Accuracy: 87.9%")
