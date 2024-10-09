import os
import pdfplumber
import docx
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from flask import Flask, render_template, request, jsonify

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

app = Flask(__name__)

def extract_text_from_pdf(file_path):
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"Error reading PDF file '{file_path}': {e}")
    return text

def extract_text_from_docx(file_path):
    text = ""
    try:
        doc = docx.Document(file_path)
        for paragraph in doc.paragraphs:
            text += paragraph.text
    except Exception as e:
        print(f"Error reading DOCX file '{file_path}': {e}")
    return text

def extract_text_from_text(file_path):
    text = ""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    except Exception as e:
        print(f"Error reading text file '{file_path}': {e}")
    return text

def extract_text(file_path):
    if file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        return extract_text_from_docx(file_path)
    elif file_path.endswith(".txt"):
        return extract_text_from_text(file_path)
    
    print(f"Error: Unsupported file type '{file_path}'")
    return ""

def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return " ".join(filtered_tokens)  # Join tokens back into a single string

def classify_cv(cv_text):
    # Placeholder classification logic - replace with your actual classification model
    # Here you can implement your logic to classify the CV into different domains
    # For example, based on keywords or a trained ML model
    if "python" in cv_text:
        return "Data Scientist"
    elif "nlp" in cv_text:
        return "Data Scientist"
    elif "machine learning" in cv_text:
        return "Data Scientist"
    elif "deep learning" in cv_text:
        return "Data Scientist"
    elif "html" in cv_text:
        return "Frontend Developer"
    elif "css" in cv_text:
        return "Frontend Developer"
    elif "javascript" in cv_text:
        return "Frontend Developer"
    elif "developer" in cv_text:
        return "Software Development"
    elif "designer" in cv_text:
        return "Graphic Designer"
    else:
        return "Other"

def match_resumes(job_description, resume_files):
    # Preprocess job description
    processed_job_desc = preprocess_text(job_description)

    resumes = []
    for resume_file in resume_files:
        resume_text = extract_text(resume_file)
        if resume_text:  # Only process if text extraction was successful
            resumes.append(preprocess_text(resume_text))

    if not resumes:
        return [], []

    # Combine job description with resumes
    documents = [processed_job_desc] + resumes

    # Apply TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Calculate cosine similarity between job description (first doc) and resumes (remaining docs)
    job_vector = tfidf_matrix[0]  # First document is job description
    resume_vectors = tfidf_matrix[1:]  # Remaining documents are resumes

    similarities = cosine_similarity(job_vector, resume_vectors)[0]

    # Get top 3 resumes
    similarities = np.array(similarities)  # Ensure similarities is a NumPy array
    top_indices = similarities.argsort()[-3:][::-1]  # Sort by similarity score and get top 3

    return top_indices, similarities

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    # Get uploaded CV
    cv_file = request.files['cv']
    
    # Save the CV temporarily to extract its content
    file_path = os.path.join("uploads", cv_file.filename)
    cv_file.save(file_path)

    # Extract text and classify the CV
    cv_text = extract_text(file_path)
    classification = classify_cv(cv_text)

    # Clean up temporary file
    os.remove(file_path)

    return jsonify({"classification": classification})

@app.route('/match', methods=['POST'])
def match():
    # Get job description from form
    job_description = request.form['job_description']
    
    # Get uploaded resumes
    resume_files = request.files.getlist('resumes')
    
    # Save the resumes temporarily to extract their content
    saved_files = []
    for resume in resume_files:
        file_path = os.path.join("uploads", resume.filename)
        resume.save(file_path)
        saved_files.append(file_path)

    # Match resumes to the job description
    top_indices, similarities = match_resumes(job_description, saved_files)

    # Clean up temporary files
    for file_path in saved_files:
        os.remove(file_path)

    # Prepare the result in JSON format
    result = {
        "top_resumes": top_indices.tolist(),
        "similarity_scores": similarities.tolist()
    }

    return jsonify(result)

if __name__ == '__main__':
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    app.run(debug=True)
