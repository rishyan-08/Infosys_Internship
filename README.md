# AI-Driven Recruitment Pipeline

An AI-based recruitment system that automates candidate screening using resumes, interview transcripts, and job descriptions.

## Overview

This project uses NLP and machine learning to:
- Preprocess and analyze candidate resumes and interview transcripts
- Extract features using TF-IDF vectorization
- Compute similarity between candidates and job requirements
- Predict candidate selection using a Random Forest classifier

## Results

- **Model**: Random Forest Classifier
- **Accuracy**: 71%
- **Precision**: 0.71
- **Recall**: 0.71
- **F1-Score**: 0.71
- **Dataset**: 3,174 samples across multiple job roles

## Project Structure

```
├── Datasets/              # Raw dataset files
├── Notebooks/             # Jupyter notebooks for exploration
├── EDA.py                 # Exploratory data analysis
├── Training.py            # Model training pipeline
├── Prediction.py          # Prediction and evaluation
├── Resume_screener.py     # Resume screening pipeline
├── data_generation.py     # Synthetic data generation using LLM
└── requirements.txt       # Python dependencies
```

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/rishyan-08/Infosys_Internship.git
   cd Infosys_Internship
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env and add your Together API key (for data generation only)
   ```

## Usage

- Run `EDA.py` for exploratory data analysis
- Run `Training.py` to train the model
- Run `Resume_screener.py` for candidate screening
- Run `Prediction.py` for predictions

## Technologies

- Python
- Scikit-learn
- NLTK
- Pandas, NumPy
- Matplotlib, Seaborn
- Together API (for data generation)

## License

This project is for educational purposes.
