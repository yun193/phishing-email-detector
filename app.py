from flask import Flask, request, jsonify
import joblib
import numpy as np
from scipy.sparse import hstack
import re
import string
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

# 載入模型（啟動時載入一次）
vectorizer = joblib.load('tfidf_vectorizer.pkl')
scaler = joblib.load('handcrafted_scaler.pkl')
model = joblib.load('phishing_detector_model.pkl')

def extract_features(email_text):
    text_vec = vectorizer.transform([email_text])
    
    url_count = len(re.findall(r'http[s]?://', email_text))
    exclamation_count = email_text.count('!')
    uppercase_ratio = sum(c.isupper() for c in email_text) / len(email_text) if len(email_text)>0 else 0
    punctuation_ratio = sum(c in string.punctuation for c in email_text) / len(email_text) if len(email_text)>0 else 0
    
    handcrafted = np.array([[url_count, exclamation_count, punctuation_ratio, uppercase_ratio]])
    handcrafted_scaled = scaler.transform(handcrafted)
    
    return hstack([text_vec, handcrafted_scaled])

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    email_text = data.get('email_text', '')
    
    features = extract_features(email_text)
    prob = model.predict_proba(features)[0][1]
    pred = model.predict(features)[0]
    
    result = {
        'is_phishing': bool(pred),
        'confidence': float(prob),
        'message': '釣魚郵件！' if pred else '正常郵件'
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)