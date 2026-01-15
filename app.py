import os
import joblib
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from sklearn.metrics.pairwise import cosine_similarity
from app.utils.file_reader import extract_text_from_stream
from app.utils.preprocessing import clean_text

app = Flask(__name__, template_folder='app/templates', static_folder='app/static')

# Configuration
UPLOAD_FOLDER = 'data/resumes'
ALLOWED_EXTENSIONS = {'pdf', 'docx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('app/models', exist_ok=True)

# Load Models
# We use a try-except block in case models aren't trained yet
try:
    model = joblib.load('app/models/model.pkl')
    vectorizer = joblib.load('app/models/vectorizer.pkl')
except:
    model = None
    vectorizer = None
    print("WARNING: Models not found. Please train the model first.")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

from app.utils.sample_jds import SAMPLE_JDS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ats')
def ats_scanner():
    return render_template('ats.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'resume_files' not in request.files:
        return redirect(request.url)
    
    files = request.files.getlist('resume_files')
    job_description = request.form.get('job_description', '')
    
    results = []
    
    # Preprocess Job Description
    cleaned_jd = clean_text(job_description)
    try:
        if vectorizer:
            jd_vector = vectorizer.transform([cleaned_jd])
        else:
            jd_vector = None
    except Exception as e:
        print(f"Error transforming JD: {e}")
        jd_vector = None

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            try:
                resume_text = extract_text_from_stream(file, filename)
                cleaned_resume = clean_text(resume_text)
            except Exception as e:
                print(f"Error reading file {filename}: {e}")
                continue
            
            # Predict Category
            category = "Unknown"
            if model and vectorizer:
                resume_vector = vectorizer.transform([cleaned_resume])
                prediction = model.predict(resume_vector)
                category = prediction[0]
                
                # Match Score against Provided JD
                if jd_vector is not None:
                    score = cosine_similarity(jd_vector, resume_vector)[0][0] * 100
                    score = round(score, 2)
                else:
                    score = 0
            else:
                score = 0
            
            results.append({
                'filename': filename,
                'category': category,
                'score': score,
                'excerpt': cleaned_resume[:200] + "..." # Preview
            })
    
    # Rank resumes by score
    results.sort(key=lambda x: x['score'], reverse=True)
    
    return render_template('result.html', results=results, job_description=job_description)

import random

@app.route('/analyze_ats', methods=['POST'])
def analyze_ats():
    if 'resume_file' not in request.files:
        return redirect('/ats')
    
    file = request.files['resume_file']
    if file.filename == '':
        return redirect('/ats')

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        try:
            resume_text = extract_text_from_stream(file, filename)
            cleaned_resume = clean_text(resume_text)
        except Exception as e:
            print(f"Error reading file {filename}: {e}")
            return "Error reading file", 500
        
        # Logic for ATS Analysis
        category = "Unknown"
        score = 0
        recommended_jd = "Not Available"
        comparisons = {}
        
        if model and vectorizer:
            resume_vector = vectorizer.transform([cleaned_resume])
            prediction = model.predict(resume_vector)
            category = prediction[0]
            
            # Get Sample JD
            recommended_jd = SAMPLE_JDS.get(category, "")
            
            if recommended_jd:
                clean_sample_jd = clean_text(recommended_jd)
                sample_jd_vector = vectorizer.transform([clean_sample_jd])
                # Calculate Similarity
                match_score = cosine_similarity(sample_jd_vector, resume_vector)[0][0] * 100
                score = round(match_score, 2)
                
                # Generate Benchmark Comparisons (Simulated)
                # varying weights to simulate different algorithms
                files_base = min(score + random.uniform(-5, 8), 98) # ChatGPT often generous on context
                files_base = max(files_base, 40)
                
                comparisons = {
                    'ChatGPT 4o': round(min(max(score + random.uniform(-4, 6), 10), 99), 1),
                    'Google ATS': round(min(max(score + random.uniform(-8, 8), 10), 99), 1),
                    'Resume Worded': round(min(max(score + random.uniform(-10, 5), 10), 99), 1), # Often stricter
                    'Enhancv': round(min(max(score + random.uniform(-5, 10), 10), 99), 1)
                }
            else:
                recommended_jd = "No standard job description available for this category yet."
                score = 0
        
        return render_template('ats_result.html', 
                               score=score, 
                               category=category, 
                               recommended_jd=recommended_jd,
                               filename=filename,
                               comparisons=comparisons)
    
    return redirect('/ats')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
