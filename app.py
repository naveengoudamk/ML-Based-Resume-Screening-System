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
            # We process directly from stream to avoid saving massive temp files if not needed,
            # but usually good to save.
            # Convert stream to text
            resume_text = extract_text_from_stream(file, filename)
            cleaned_resume = clean_text(resume_text)
            
            # Predict Category
            category = "Unknown"
            if model and vectorizer:
                resume_vector = vectorizer.transform([cleaned_resume])
                prediction = model.predict(resume_vector)
                category = prediction[0]
                
                # Match Score
                if jd_vector is not None:
                    # Cosine Similarity returns a matrix [[score]]
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

if __name__ == '__main__':
    app.run(debug=True, port=5000)
