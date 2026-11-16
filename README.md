# Coffee Bean Image Classifier ☕

大學畢業專題：咖啡豆瑕疵辨識網頁系統。

使用者上傳咖啡豆照片後，系統會自動去背、以 CNN 模型判斷每顆豆子是「正常」或「瑕疵」，並統計瑕疵率。

## 功能

- 拖曳上傳多張照片（Dropzone.js），自動擋下重複檔案與非圖片格式
- 以 [rembg](https://github.com/danielgatis/rembg) 自動去背
- Keras CNN 模型（輸入 256×256）分類正常豆 / 瑕疵豆
- 顯示每張照片的判斷結果、正常與瑕疵豆數量、瑕疵率

## 技術

Python · Flask · TensorFlow / Keras · rembg · Pillow · Dropzone.js

## 專案結構

```
senior-project/
├── app.py              # Flask 主程式（上傳、去背、預測）
├── modelold.h5         # 舊版模型（無 BatchNormalization）
├── templates/
│   └── index.html      # 前端頁面
└── static/
    ├── dropzone/       # 上傳元件
    └── images/         # 範例 / 去背後的豆子照片
```

## 執行方式

```bash
pip install flask tensorflow rembg pillow numpy h5py
mkdir -p static/imagesPrediction   # 上傳暫存資料夾
python app.py                      # http://localhost:3000
```

> ⚠️ `app.py` 預設讀取 `model.h5`（含 BatchNormalization 的新版模型），但 repo 中只有 `modelold.h5`。
> 若沒有新版模型，可將 `app.py` 中的載入改為註解掉的那行 `load_model('modelold.h5')`。
