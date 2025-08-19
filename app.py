# import pandas as pd
# from flask import Flask, render_template, request
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
# import string
# import pickle # Import the pickle library

# # Download necessary NLTK data (only run once when deploying)
# # New code:
# try:
#     nltk.data.find('corpora/stopwords')
# except Exception: # Catch a more general Exception, or use specific NLTK classes if you prefer
#     nltk.download('stopwords')
# try:
#     nltk.data.find('tokenizers/punkt')
# except Exception: # Catch a more general Exception
#     nltk.download('punkt')

# app = Flask(__name__)


# # Add these lines temporarily, then remove them after debugging
# # To prevent the program from exiting if files aren't found, you might comment out the exit()
# # or just run this snippet separately.
# try:
#    # Inside app.py, change these lines:
#     model = pickle.load(open('model.pkl', 'rb'))
#     tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
# except FileNotFoundError:
#     print("Error: model.pkl or vectorize.pkl not found.")
#     print("Please ensure 'model.pkl' (your trained ML model) and 'vectorize.pkl' (your fitted TF-IDF vectorizer) are in the same directory as app.py.")
#     # Exit or handle this error gracefully in a production app
#     exit()

# # Text Transformation Function (from your notebook's logic)
# ps = PorterStemmer()
# def transform_text(text):
#     text = text.lower()
#     text = nltk.word_tokenize(text)

#     y = []
#     for i in text:
#         if i.isalnum():
#             y.append(i)

#     text = y[:]
#     y.clear()
#     for i in text:
#         if i not in stopwords.words('english') and i not in string.punctuation:
#             y.append(i)

#     text = y[:]
#     y.clear()
#     for i in text:
#         y.append(ps.stem(i))

#     return " ".join(y)

# # --- Flask Routes ---
# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         sms_message = request.form['sms_message']
        
#         # 1. Preprocess the input text
#         transformed_sms = transform_text(sms_message)
        
#         # 2. Vectorize the preprocessed text using the loaded TF-IDF vectorizer
#         vector_input = tfidf.transform([transformed_sms])
        
#         # 3. Make prediction using the loaded model
#         result = model.predict(vector_input)[0]
        
#         output = "Spam" if result == 1 else "Not Spam"
        
#         return render_template('index.html', prediction_text=f'The message is: {output}')

# if __name__ == '__main__':
#     app.run(debug=True)

import pandas as pd
from flask import Flask, render_template, request
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import pickle # Import the pickle library

# --- START NLTK DOWNLOAD BLOCK ---
import ssl # Import ssl for handling potential certificate issues

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download necessary NLTK data if not already present
# This checks and downloads 'punkt_tab', 'stopwords', and 'punkt'
try:
    nltk.data.find('tokenizers/punkt_tab')
except nltk.downloader.DownloadError:
    print("NLTK 'punkt_tab' not found, attempting to download...")
    nltk.download('punkt_tab')
    
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    print("NLTK 'stopwords' not found, attempting to download...")
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("NLTK 'punkt' not found, attempting to download...")
    nltk.download('punkt')

print("NLTK data check complete.")
# --- END NLTK DOWNLOAD BLOCK ---


app = Flask(__name__)

# Add these lines temporarily, then remove them after debugging
# To prevent the program from exiting if files aren't found, you might comment out the exit()
# or just run this snippet separately.
try:
    # Inside app.py, change these lines:
    model = pickle.load(open('model.pkl', 'rb'))
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
except FileNotFoundError:
    print("Error: model.pkl or vectorize.pkl not found.")
    print("Please ensure 'model.pkl' (your trained ML model) and 'vectorize.pkl' (your fitted TF-IDF vectorizer) are in the same directory as app.py.")
    # Exit or handle this error gracefully in a production app
    exit()

# Text Transformation Function (from your notebook's logic)
ps = PorterStemmer()
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text) # This is where punkt_tab is needed indirectly for sent_tokenize that word_tokenize might call

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# --- Flask Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        sms_message = request.form['sms_message']
        
        # 1. Preprocess the input text
        transformed_sms = transform_text(sms_message)
        
        # 2. Vectorize the preprocessed text using the loaded TF-IDF vectorizer
        vector_input = tfidf.transform([transformed_sms])
        
        # 3. Make prediction using the loaded model
        result = model.predict(vector_input)[0]
        
        output = "Spam" if result == 1 else "Not Spam"
        
        return render_template('index.html', prediction_text=f'The message is: {output}')

if __name__ == '__main__':
    app.run(debug=True)