import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
from app.utils.preprocessing import clean_text

# 1. Create a Synthetic Dataset
# In a real project, this would be loaded from a CSV/Kaggle dataset
print("Creating synthetic dataset...")
data = {
    'Resume_Text': [
        "Java Developer, Spring Boot, Hibernate, SQL, Microservices, JUnit, Maven, REST API", # Java Dev
        "Python Developer, Django, Flask, Pandas, NumPy, SQL, API Development", # Python Dev
        "Data Scientist, Machine Learning, Deep Learning, NLP, TensorFlow, Keras, Python, Statistics", # Data Science
        "Web Designer, HTML, CSS, JavaScript, React, Bootstrap, Photoshop, UI/UX", # Web Designer
        "HR Manager, Recruitment, Employee Relations, Payroll, Compliance, Communication", # HR
        "Sales Executive, Marketing, CRM, Lead Generation, Negotiation, Business Development", # Sales
        "Android Developer, Kotlin, Java, XML, Android SDK, Dagger, Retrofit", # Android
        "DevOps Engineer, AWS, Docker, Kubernetes, Jenkins, Linux, CI/CD", # DevOps
        
        # Variations
        "Expert in Java and Spring framework, built scalable microservices.",
        "Experienced Data Analyst with strong Python and ML skills.",
        "Front-end developer proficient in React.js and Tailwind CSS.",
        "Human Resources professional with 5 years in talent acquisition.",
        
        # More specific data for better training
        "Implemented RESTful web services using Java and Spring Boot.",
        "Developed neural networks for image classification using PyTorch.",
        "Designed responsive websites using HTML5 and CSS3.",
        "Managed end-to-end recruitment lifecycle for technical roles."
    ],
    'Category': [
        'Java Developer', 'Python Developer', 'Data Science', 'Web Designing', 'HR', 'Sales', 'Android Developer', 'DevOps',
        'Java Developer', 'Data Science', 'Web Designing', 'HR',
        'Java Developer', 'Data Science', 'Web Designing', 'HR'
    ]
}

df = pd.read_csv('data/resumes_dataset.csv') if os.path.exists('data/resumes_dataset.csv') else pd.DataFrame(data)

# If using synthetic data, let's duplicate it to have enough for a split if needed, 
# but for a demo, we will just train on all of it if small.
if len(df) < 50:
    df = pd.concat([df]*5, ignore_index=True)

print(f"Dataset size: {len(df)}")

# 2. Preprocessing
print("Cleaning text...")
df['Cleaned_Resume'] = df['Resume_Text'].apply(clean_text)

# 3. Feature Extraction
print("Vectorizing...")
tfidf = TfidfVectorizer(max_features=2000)
X = tfidf.fit_transform(df['Cleaned_Resume'])
y = df['Category']

# 4. Model Training
print("Training model...")
# Using OneVsRest with SVC for multi-class classification
clf = OneVsRestClassifier(SVC(kernel='linear', probability=True))
clf.fit(X, y)

# 5. Evaluation
print(f"Model Accuracy on training set: {clf.score(X, y):.4f}")

# 6. Save Artifacts
print("Saving model and vectorizer...")
os.makedirs('app/models', exist_ok=True)
joblib.dump(clf, 'app/models/model.pkl')
joblib.dump(tfidf, 'app/models/vectorizer.pkl')

print("Done! Models saved to app/models/")
