import streamlit as st
import joblib
import numpy as np
from scipy.sparse import hstack
import re
import string
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# 載入模型
@st.cache_resource  # 只載入一次
def load_model():
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    scaler = joblib.load('handcrafted_scaler.pkl')
    model = joblib.load('phishing_detector_model.pkl')
    return vectorizer, scaler, model

vectorizer, scaler, model = load_model()

# 特徵提取函式
def extract_features(email_text):
    text_vec = vectorizer.transform([email_text])
    
    url_count = len(re.findall(r'http[s]?://', email_text))
    exclamation_count = email_text.count('!')
    uppercase_ratio = sum(c.isupper() for c in email_text) / len(email_text) if len(email_text) > 0 else 0
    punctuation_ratio = sum(c in string.punctuation for c in email_text) / len(email_text) if len(email_text) > 0 else 0
    
    handcrafted = np.array([[url_count, exclamation_count, punctuation_ratio, uppercase_ratio]])
    handcrafted_scaled = scaler.transform(handcrafted)
    
    return hstack([text_vec, handcrafted_scaled])

# Streamlit 介面
st.title("🎣 釣魚郵件偵測系統")
st.markdown("### 貼上郵件內容，AI 立刻告訴你是否為釣魚郵件")

st.info("💡 資安小知識：釣魚郵件常有誘導詞（如 click、urgent）、多個 URL、大量驚嘆號！")

email_text = st.text_area("請貼上郵件內容（包含主旨與本文）：", height=200)

if st.button("偵測"):
    if email_text.strip():
        with st.spinner("AI 正在分析中..."):
            features = extract_features(email_text)
            prob = model.predict_proba(features)[0][1]
            pred = model.predict(features)[0]
        
        if pred == 1:
            st.error(f"⚠️  檢測為釣魚郵件！")
            st.warning(f"信心分數：{prob:.2%}（越高越確定）")
        else:
            st.success(f"✅ 檢測為正常郵件")
            st.info(f"釣魚機率僅：{prob:.2%}")
        
        # 加資安解釋
        st.markdown("### 資安分析提示")
        if "click" in email_text.lower() or "urgent" in email_text.lower():
            st.warning("⚡ 偵測到誘導詞（如 click、urgent），這是釣魚常見手法！")
        if len(re.findall(r'http[s]?://', email_text)) > 1:
            st.warning("🔗 郵件含多個連結，可能是惡意 URL！")
    else:
        st.warning("請輸入郵件內容！")

st.markdown("---")
st.caption("本系統使用 TF-IDF + 手工特徵 + Logistic Regression 模型訓練，資料來自 Enron/CEAS/Phishing Email 資料集")