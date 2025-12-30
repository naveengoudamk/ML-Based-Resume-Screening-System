# ML-Based Resume Screening System

This is a complete end-to-end Machine Learning project that screens and ranks resumes based on a given job description.

## 🚀 Features
- **Resume Parsing**: Extracts text from PDF and DOCX files.
- **NLP Preprocessing**: Cleans text (stopwords, lemmatization).
- **ML Classification**: Categorizes resumes into roles (e.g., Java Developer, Data Scientist) using SVM (Support Vector Machine).
- **Ranking System**: Uses **Cosine Similarity** to match resumes against the Job Description and rank them by relevance percentage.
- **Premium UI**: Modern, glassmorphism-based web interface built with Flask.

## 📂 Project Structure
```
/app
  /models       # Trained ML models (model.pkl, vectorizer.pkl)
  /static       # CSS and database
  /templates    # HTML files
  /utils        # Helper scripts (parsing, cleaning)
  app.py        # Main Flask Application
train_model.py  # Script to train the ML model
requirements.txt
```

## 🛠️ How to Run

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model** (First time only)
   ```bash
   python3 train_model.py
   ```

3. **Run the Web App**
   ```bash
   python3 app.py
   ```

4. **Open in Browser**
   Go to `http://127.0.0.1:5000`

## 🧪 Testing Locally
1. Run the app.
2. Enter a Job Description (e.g., "Looking for a Data Scientist with Python and NLP experience.").
3. Upload sample resumes (PDF or DOCX).
4. See the ranked results!
