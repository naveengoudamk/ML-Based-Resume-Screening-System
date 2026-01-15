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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'resume_files' not in request.files:
        return redirect(request.url)
    
    files = request.files.getlist('resume_files')
    # strip to ensure empty string if whitespace only
    user_job_description = request.form.get('job_description', '').strip()
    
    results = []
    
    # Preprocess User Job Description if provided
    user_jd_vector = None
    if user_job_description and vectorizer:
        try:
            cleaned_user_jd = clean_text(user_job_description)
            user_jd_vector = vectorizer.transform([cleaned_user_jd])
        except Exception as e:
            print(f"Error transforming User JD: {e}")

    # Import Sample JDs here or at top level. doing here to avoid circulars if any, though top is fine.
    from app.utils.sample_jds import SAMPLE_JDS

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Convert stream to text
            try:
                resume_text = extract_text_from_stream(file, filename)
                cleaned_resume = clean_text(resume_text)
            except Exception as e:
                print(f"Error reading file {filename}: {e}")
                continue
            
            # Predict Category
            category = "Unknown"
            resume_vector = None
            if model and vectorizer:
                resume_vector = vectorizer.transform([cleaned_resume])
                prediction = model.predict(resume_vector)
                category = prediction[0]
                
            # Determine Scoring Target (User JD vs Recommended/Sample JD)
            score = 0
            score_type = "ATS Score" # Default label
            
            # We always find a recommended JD based on prediction
            recommended_jd = SAMPLE_JDS.get(category, "No sample description available for this category.")
            
            if resume_vector is not None:
                if user_jd_vector is not None:
                    # Case 1: Score against User Provided JD
                    match_score = cosine_similarity(user_jd_vector, resume_vector)[0][0] * 100
                    score = round(match_score, 2)
                    score_type = "ATS Match (vs Job)"
                else:
                    # Case 2: Score against Sample JD (Self-Match / Validity)
                    # We need to vectorize the sample JD
                    if recommended_jd and vectorizer:
                        clean_sample_jd = clean_text(recommended_jd)
                        sample_jd_vector = vectorizer.transform([clean_sample_jd])
                        match_score = cosine_similarity(sample_jd_vector, resume_vector)[0][0] * 100
                        score = round(match_score, 2)
                        score_type = f"Profile Strength ({category})"
            
            results.append({
                'filename': filename,
                'category': category,
                'score': score,
                'score_type': score_type,
                'recommended_jd': recommended_jd,
                'excerpt': cleaned_resume[:200] + "..." 
            })
    
    # Rank resumes by score
    results.sort(key=lambda x: x['score'], reverse=True)
    
    return render_template('result.html', results=results, job_description=user_job_description)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
