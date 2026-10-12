1. 確認 butterfly_dataset 數據集有在此目錄內（butterfly_dataset 檔案太大放不進去）

2. 在此目錄下打開終端機建立虛擬環境
python -m venv .venv

3. 啟動虛擬環境
.venv\Scripts\activate

4. 安裝套件
pip install -r requirements.txt

5. 執行訓練模型的程式並得到 Test_Dataset.csv
python .\train.py