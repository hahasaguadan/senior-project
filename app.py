from flask import Flask, render_template, request, jsonify, redirect, url_for, abort, send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.python.keras.models import load_model 
from tensorflow.keras.preprocessing import image # 我的電腦中間不加python才可以正常運行
# from keras.api._v2.keras.preprocessing import image
from tensorflow.keras.layers import BatchNormalization
import numpy as np
import h5py
import os
from rembg import remove
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = 'static/imagesPrediction' # 暫存照片資料夾路徑
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER # 取得伺服器目前路徑
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制大小 16MB
  
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg']) #允許的附檔名
def allowed_file(filename): # 防止其他類型檔案上傳
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# count = []

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html') # 主頁面以indexhtml為主


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    data = []
    store = [] # 存放去背前照片檔案路徑
    GoodCount = 0
    BadCount = 0
    deficientRate = 0
    predImagePath = os.listdir('static/imagesPrediction')
    for f in predImagePath: 
        image_path = UPLOAD_FOLDER + '/' + f # 設定圖片存檔路徑
        store.append(image_path)
        newImagesName = BK_remove(image_path, f) # 去背
        newPath = 'static/images/'+newImagesName # 存去背後的照片到照片資料夾
        #classifier = load_model('modelold.h5') # 用模型預測豆子(無BN)
        classifier = load_model('model.h5',custom_objects={'BatchNormalization': BatchNormalization}) # 用模型預測豆子(有BN)
        test_image = image.load_img(newPath, target_size=(256,256))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image / 255
        result = classifier.predict(test_image)
        if result[0][0] > 0.01:
            prediction = 'Normal'
            GoodCount += 1
        else:
            prediction = 'Defective'
            BadCount += 1
        print(result[0][0])
        data.append([newImagesName, prediction])
    if BadCount != 0 or GoodCount != 0 :
        deficientRate = BadCount / (BadCount + GoodCount)
        deficientRate = round(deficientRate, 2)
    for f in store: # 刪除去背前的照片
        os.remove(f)
    return render_template('index.html', good=GoodCount, bad=BadCount, rate=deficientRate, data=data) # 回傳結果

@app.route("/upload",methods=["POST","GET"])
def upload():
    if request.method == 'POST':
        file = request.files['file'] # 取得圖片
        predImagePath = os.listdir(UPLOAD_FOLDER)
        for f in predImagePath : 
            if f == file.filename :
                print('same image!!')
                return '重複的照片!', 403  # return the error message, with a proper 4XX code

        # secure_filename()將會對傳入的文件路徑進行處理，將其中的路徑符號“/”用下劃線代替，且去掉中文
        filename = secure_filename(file.filename) 
        #確認有檔案且上傳文件合法
        if file and allowed_file(file.filename):
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename)) # 檔案路徑與檔名結合
            print('File successfully uploaded ' + file.filename + ' to the file!')
        else:
           print('Invalid Uplaod only png, jpg, jpeg') 
        msg = 'Success Uplaod' 

    return jsonify(msg)

@app.route('/deletefile',methods=["POST","GET"])
def delete_file():
    if request.method == 'POST':
        print(request.form['duplicate'])
        if request.form['duplicate'] == '1' : # 判斷是否為重複上傳
            return render_template('index.html')
        filename = request.form['name']
        predImagePath = os.listdir(UPLOAD_FOLDER)
        if predImagePath==[] : 
            return render_template('index.html')
        for f in predImagePath:
            if f == filename : # 這張照片在資料夾裡重複了
                file_path = UPLOAD_FOLDER + '/' + filename 
                os.remove(file_path) # 刪除重複
                return render_template('index.html')
    return render_template('index.html')

def BK_remove(inputPath, filename): # 去背轉png傳回照片名稱
    filename = filename[:filename.index('.')]
    inputs = Image.open(inputPath)
    output = remove(inputs)
    name = filename+'.png'
    output.save('static/images/'+name)
    return name

if __name__=="__main__": 
    app.run(port=3000, debug=True) 