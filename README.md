# phishing-email-detector

一個基於機器學習的**釣魚郵件偵測系統**，結合 TF-IDF 文字特徵與資安手工特徵（URL 數量、驚嘆號、標點比率、大寫比率），使用 Logistic Regression 模型達成 **99% 準確率** 與 **98.3% 釣魚召回率**。

本專案同時達到兩個學習目標：
- **資安知識**：深入理解釣魚郵件常見攻擊手法（如偽裝新聞、誘導點擊、製造緊急感）
- **AI 知識**：完整走過機器學習專案流程（資料收集 → 前處理 → EDA → 特徵工程 → 模型訓練 → 部署）

## 📊 使用資料集

本專案合併以下公開資料集進行訓練：
- Enron Email Dataset：https://www.kaggle.com/datasets/wcukierski/enron-email-dataset
- CEAS 2008 Phishing Corpus：https://www.kaggle.com/datasets/subhajournal/phishingemails
- Phishing Email Dataset：https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset/data

> 注意：原始資料檔案過大，未上傳至 GitHub。執行 `data.py` 前請自行下載以上資料集並放置正確路徑。

## 🚀 快速開始
### 1. 安裝依賴
```bash
pip install pandas scikit-learn seaborn nltk joblib fastapi uvicorn scipy
```
### 2. 安裝依賴執行資料前處理（首次執行需下載原始 CSV）
```bash
python data.py
```
### 3. 啟動 API 服務
```bash
uvicorn app_fastapi:app --reload
```

### 4.測試模型
開啟瀏覽器訪問：
<br>http://127.0.0.1:8000 → 歡迎頁面  
http://127.0.0.1:8000/docs → 互動式 API 文件（Swagger UI）</br>

在 /docs 頁面測試 /predict 端點，範例輸入：
```json
{
  "email_text": "Urgent! Your account will be suspended. Click here to verify: http://fake-bank.com"
}
```
預期回傳：
```json
{
  "is_phishing": true,
  "confidence": 0.9829,
  "message": "⚠️ 釣魚郵件！",
  "detail": "釣魚機率：98.29%"
}
```

