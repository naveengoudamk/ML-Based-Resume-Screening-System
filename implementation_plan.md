# Implementation Plan: ML-Based Resume Screening System

Based on the detailed explanation provided, I will implement the Resume Screening System. This system will automatically screen and rank resumes against job descriptions using NLP and Machine Learning.

## 1. Project Setup
- [ ] Initialize Python environment and `requirements.txt`
- [ ] Create project structure (Flask backend, templates, static files, utils).

## 2. ML Pipeline Components
- [ ] **Text Extraction (`utils/file_reader.py`)**: Implement functions to extract text from PDF (using `PyPDF2`/`pdfplumber`) and DOCX (`python-docx`).
- [ ] **Preprocessing (`utils/preprocessing.py`)**: Implement NLP cleaning (lowercase, remove punctuation, stopwords, lemmatization).
- [ ] **Feature Extraction & Model Training (`train_model.py`)**: 
    - Create a synthetic dataset (or use a small sample) to train a TF-IDF Vectorizer and a Classifier (Logistic Regression/SVM).
    - Save the model and vectorizer as `.pkl` files.

## 3. Web Application (Flask)
- [ ] **Backend (`app.py`)**:
    - Route for Home/Upload page.
    - Route for processing uploads and displaying results.
    - Logic to match Resume vs Job Description (Cosine Similarity).
- [ ] **Frontend (`templates/` & `static/`)**:
    - **Design**: Premium, modern aesthetics with glassmorphism and smooth animations.
    - **Input**: Resume upload (Drag & Drop) and Job Description text area.
    - **Output**: Dashboard showing matched score, extracted skills, and ranking.

## 4. Evaluation & Testing
- [ ] Verify PDF/DOCX parsing.
- [ ] Test matching logic with sample resumes.
- [ ] Ensure UI is responsive and visually appealing.

## 5. Deployment Preparation
- [ ] Add `README.md` with instructions on how to run.
