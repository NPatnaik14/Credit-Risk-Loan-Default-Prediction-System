# AI Resume Screening System (ATS Simulation)

An end-to-end Intelligent Resume Screening System built with Python, NLP, and Streamlit. This application simulates a real-world Applicant Tracking System (ATS) by ranking resumes based on their similarity to a specific job description.

## 🚀 Project Overview
In modern recruitment, HR teams receive hundreds of resumes for a single position. Manually screening these is time-consuming and prone to bias. This system automates the initial screening process by using Natural Language Processing (NLP) to mathematically compare resumes against job requirements.

## 🧠 System Architecture
The system consists of two main layers:

1.  **Backend (ML Pipeline):**
    *   **Text Extraction:** Uses `pdfplumber` to extract raw text from PDF resumes.
    *   **Preprocessing:** Tokenization, lowercasing, punctuation removal, stopword removal, and Lemmatization (using NLTK).
    *   **Vectorization:** Converts text into numerical vectors using **TF-IDF (Term Frequency-Inverse Document Frequency)**.
    *   **Similarity Computation:** Calculates the **Cosine Similarity** between resume vectors and the job description vector.
2.  **Frontend (Streamlit UI):**
    *   A clean web interface for uploading multiple resumes and pasting job descriptions.
    *   Real-time processing with a visual ranking table and a similarity score bar chart.

## 🛠 NLP Techniques Used
*   **Lemmatization:** Unlike stemming (which just chops off word ends), lemmatization reduces words to their actual dictionary root (e.g., "running" becomes "run"), preserving context.
*   **TF-IDF:** Assigns weight to words based on their importance in a document relative to a corpus. This helps emphasize rare but meaningful keywords (like "Python" or "Kubernetes") over common words.
*   **Cosine Similarity:** Measures the cosine of the angle between two vectors. It is effective for text because it measures orientation (context) rather than magnitude (length of resume).

## 📊 How Ranking Works
1.  Both resumes and the job description are cleaned and converted into a shared vector space.
2.  The "distance" between each resume and the JD is calculated.
3.  A score of `1.0` (100%) indicates a perfect match, while `0.0` indicates no overlapping relevant terms.
4.  Resumes are sorted in descending order of their scores.

## 🛠 How to Run Locally

### 1. Prerequisites
Ensure you have Python 3.8+ installed.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the App
```bash
streamlit run streamlit_app.py
```

## 🔮 Future Upgrades
*   **Deep Learning:** Replace TF-IDF with **BERT (Bidirectional Encoder Representations from Transformers)** for better semantic understanding.
*   **Skill Extraction:** Implement Named Entity Recognition (NER) to specifically extract skills, education, and experience levels.
*   **Section Parsing:** Better extraction of specific sections like "Work Experience" and "Projects".
*   **Deployment:** Containerize using Docker and deploy to AWS/GCP/Streamlit Cloud.

---
Built as a professional ML product simulation.
