# ButterViT：基於 Vision Transformer 的蝴蝶影像辨識

> 以預訓練 ViT 微調的蝴蝶影像二元分類模型。在 427 張外部測試圖上達到 **99.06% 準確率**，**Sensitivity 100%**。

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Transformers-FFD21E?logo=huggingface&logoColor=black)

---
### 蝴蝶雜交種辨識，雜交種與非雜交種的舉例
<img width="1915" height="1079" alt="image" src="https://github.com/user-attachments/assets/d4d7c54c-e609-45be-8cac-4da5409176e9" />

## 成果

外部測試集共 427 張（正類 402 張、負類 25 張），與 `Perfect_Answer.csv` 比對的結果：

| 指標 | 結果 |
|---|---|
| **Accuracy** | **99.06%** |
| **TPR（Sensitivity）** | **100%**（402 / 402） |
| **TNR（Specificity）** | **84%**（21 / 25） |
| TP / TN / FP / FN | 402 / 21 / 4 / 0 |

---

## 重點一覽

| 項目 | 內容 |
|---|---|
| 任務 | 影像二元分類（1 = Positive，0 = Negative） |
| 骨幹模型 | `google/vit-base-patch16-224-in21k` |
| 分類頭 | Linear(768→384) → ReLU → Dropout(0.2) → Linear(384→2) |
| 資料切分 | **Train 70% / Val 15% / Test 15%**（多種比例實驗後效果最好） |
| 資料增強 | Resize 224、水平翻轉、±10° 隨機旋轉、ImageNet 正規化 |
| 訓練設定 | AdamW（lr 2e-5、weight decay 1e-2）、Batch 16、10 Epochs |
| 模型選擇 | 保留驗證集準確率最高的那個 epoch |
| 程式語言 | Python |

---

## 專案結構

```
Butterfly_project/
├── train.py             # 主程式：訓練、內部測試、對外部測試集預測
├── dataset.py           # 資料讀取、70/15/15 切分、資料增強
├── model.py             # ViT 骨幹 + MLP 分類頭 + optimizer
├── evaluate.py          # 預測結果與標準答案比對（Accuracy / TPR / TNR）
├── Perfect_Answer.csv   # 外部測試集標準答案
└── requirements.txt     # 需要的套件
```

資料集太大，沒有放進 repo，請自行放在專案根目錄：

```
butterfly_dataset/
├── train/
│   ├── <類別0>/   *.jpg
│   └── <類別1>/   *.jpg
└── test/          *.jpg（無標籤）
```

---

## 快速開始

```bash
# 1. 建立並啟動虛擬環境（Windows）
python -m venv .venv
.venv\Scripts\activate

# 2. 安裝套件
pip install -r requirements.txt

# 3. 訓練，並對外部測試集預測（會產生 vit_cls_best.pth 和 Test_Dataset.csv）
python train.py

# 4. 和標準答案比對
python evaluate.py
```

---

## 使用技術

- **語言**：Python
- **深度學習**：PyTorch、torchvision
- **預訓練模型**：Hugging Face Transformers（ViT）
- **資料處理**：pandas、Pillow
- **其他**：tqdm（進度條）
