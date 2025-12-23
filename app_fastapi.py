from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from scipy.sparse import hstack
import re
import string
import warnings

# 忽略無害的 scaler 警告
warnings.filterwarnings("ignore", category=UserWarning)

# 初始化 FastAPI
app = FastAPI(
    title="釣魚郵件偵測系統",
    description="使用 TF-IDF + 手工特徵的機器學習模型，偵測郵件是否為釣魚郵件（資安應用）",
    version="1.0.0"
)

# 載入訓練好的模型與工具（啟動時載入一次）
print("正在載入模型與工具...")
vectorizer = joblib.load('tfidf_vectorizer.pkl')
scaler = joblib.load('handcrafted_scaler.pkl')
model = joblib.load('phishing_detector_model.pkl')
print("模型載入完成！API 準備就緒")

# 定義輸入格式（Pydantic 會自動驗證）
class EmailRequest(BaseModel):
    email_text: str

# 特徵提取函式（跟之前完全一樣）
def extract_features(email_text: str):
    # TF-IDF
    text_vec = vectorizer.transform([email_text])
    
    # 手工特徵
    url_count = len(re.findall(r'http[s]?://', email_text))
    exclamation_count = email_text.count('!')
    uppercase_ratio = sum(c.isupper() for c in email_text) / len(email_text) if len(email_text) > 0 else 0
    punctuation_ratio = sum(c in string.punctuation for c in email_text) / len(email_text) if len(email_text) > 0 else 0
    
    handcrafted = np.array([[url_count, exclamation_count, punctuation_ratio, uppercase_ratio]])
    handcrafted_scaled = scaler.transform(handcrafted)
    
    # 合併
    return hstack([text_vec, handcrafted_scaled])

# 根路徑（打開瀏覽器會看到歡迎訊息）
@app.get("/")
def home():
    return {
        "message": "釣魚郵件偵測 API 已上線！",
        "docs": "請訪問 /docs 進行互動測試（Swagger UI）",
        "redoc": "或訪問 /redoc 看另一種文件樣式"
    }

# 預測端點
@app.post("/predict")
def predict(email: EmailRequest):
    try:
        features = extract_features(email.email_text)
        prob = model.predict_proba(features)[0][1]  # 釣魚機率
        pred = model.predict(features)[0]
        
        return {
            "is_phishing": bool(pred),
            "confidence": round(float(prob), 4),
            "message": "⚠️ 釣魚郵件！" if pred else "✅ 正常郵件",
            "detail": f"釣魚機率：{prob:.2%}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))