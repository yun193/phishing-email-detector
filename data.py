import pandas as pd #解析excel套件
import os
import re

# ----------------------------- 使用者設定區 -----------------------------
# 在這裡修改你的 CSV 檔案路徑！
# 可以是絕對路徑（如 "C:/Users/你的名字/Downloads/emails.csv"）
# 也可以是相對路徑（如 "data/emails.csv"，表示目前資料夾下的 data 子資料夾）

csv_files = {
    "enron": {
        "path": "D:\課外練習\side-projrct\phishing_email\資料集\emails.csv",                  # <--- 修改這裡
        "type": "enron"
    },
    "ceas": {
        "path": "D:\課外練習\side-projrct\phishing_email\資料集\CEAS_08.csv",                 # <--- 修改這裡
        "type": "standard"
    },
    "phishing_email": {
        "path": "D:\課外練習\side-projrct\phishing_email\資料集\Phishing_Email.csv",          # <--- 修改這裡
        "type": "phishing_type"
    }
}

# 輸出檔案名稱與路徑（也可以改成其他位置）
output_file = "D:\課外練習\side-projrct\phishing_email\資料集\combined_phishing_dataset_clean.csv"

# 平衡設定：最終希望釣魚與正常郵件各約多少筆（避免資料太多跑不動）
target_count_per_class = 30000   # 每類保留最多 3 萬筆（可自行調整）

min_text_length = 20             # 郵件內容最短長度（太短的過濾掉）
# --------------------------------------------------------------------

# 工具函式：從 Enron 的 message 提取正文
def extract_body(message):
    if pd.isna(message):
        return ""
    parts = str(message).split('\n\n', 1)  # 以第一個空行分隔 header 和 body
    body = parts[1] if len(parts) > 1 else str(message)
    body = re.sub(r'\s+', ' ', body)      # 所有空白（換行、tab）變成單一空格
    return body.strip()

# 主函式：讀取並標準化單一 CSV
def load_csv_file(file_info):
    file_path = file_info["path"]
    file_type = file_info["type"]
    
    print(f"\n正在讀取：{file_path}")
    
    # 檢查檔案是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"檔案不存在！請檢查路徑：{file_path}")
    
    # 讀取 CSV
    df = pd.read_csv(file_path, low_memory=False)
    print(f"原始筆數：{len(df)} 欄位：{list(df.columns)}")
    
    # 根據不同來源處理
    if file_type == "enron":
        # 修正標籤：只有檔名含 'spam' 才是釣魚郵件，其餘都是正常
        df['label'] = df['file'].apply(lambda x: 1 if 'spam' in str(x).lower() else 0)
        df['text'] = df['message'].apply(extract_body)
        
    elif file_type == "phishing_type":
        # Phishing_Email.csv 使用 'Email Type' 當標籤
        df['text'] = df['Email Text'].fillna("").astype(str)
        mapping = {'Phishing Email': 1, 'Safe Email': 0}
        df['label'] = df['Email Type'].map(mapping)
        # 如果有未知值，預設為 0（安全起見）
        df['label'] = df['label'].fillna(0).astype(int)
        
    elif file_type == "standard":
        # CEAS_08.csv 等標準格式
        df['text'] = df['body'].fillna("").astype(str)
        df['label'] = df['label'].astype(int)
    
    else:
        raise ValueError(f"未知的檔案類型：{file_type}")
    
    # 最終清理
    df = df[['text', 'label']].copy()
    df = df.dropna(subset=['label'])
    df = df[df['text'].str.len() >= min_text_length]
    df = df.drop_duplicates(subset=['text'])
    
    phishing_count = (df['label'] == 1).sum()
    ham_count = (df['label'] == 0).sum()
    print(f"處理完成 → 釣魚郵件：{phishing_count} 筆，正常郵件：{ham_count} 筆")
    
    return df

# ----------------------------- 主程式開始 -----------------------------
print("開始合併多個釣魚郵件資料集...\n")

dataframes = []

# 逐一讀取每個 CSV
for name, info in csv_files.items():
    try:
        df = load_csv_file(info)
        dataframes.append(df)
    except Exception as e:
        print(f"讀取 {name} 失敗：{e}")

# 合併所有資料
combined_df = pd.concat(dataframes, ignore_index=True)
print(f"\n所有檔案合併完成！總筆數：{len(combined_df)}")
print(f"釣魚郵件：{(combined_df['label']==1).sum()} 筆")
print(f"正常郵件：{(combined_df['label']==0).sum()} 筆")

# ----------------------------- 平衡資料 -----------------------------
print("\n正在平衡資料集（每類約 {} 筆）...".format(target_count_per_class))

phishing_df = combined_df[combined_df['label'] == 1]
ham_df = combined_df[combined_df['label'] == 0]

# 隨機抽樣（不夠就全部用）
phishing_sample = phishing_df.sample(n=min(target_count_per_class, len(phishing_df)), random_state=42)
ham_sample = ham_df.sample(n=min(target_count_per_class, len(ham_df)), random_state=42)

# 合併並洗牌
final_df = pd.concat([phishing_sample, ham_sample])
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"平衡完成！最終資料集：{len(final_df)} 筆")
print(f"釣魚郵件：{(final_df['label']==1).sum()} 筆 ({(final_df['label']==1).sum()/len(final_df)*100:.1f}%)")
print(f"正常郵件：{(final_df['label']==0).sum()} 筆")

# ----------------------------- 儲存結果 -----------------------------
final_df.to_csv(output_file, index=False, encoding='utf-8')
print(f"\n資料已成功儲存至：{output_file}")

# 顯示前幾筆讓你確認
print("\n前 10 筆資料預覽：")
print(final_df[['text', 'label']].head(10))