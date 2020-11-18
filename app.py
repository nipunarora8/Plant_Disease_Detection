from flask import Flask, redirect, url_for, request, render_template,send_from_directory
from keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import os,cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from keras.models import *
app = Flask(__name__)

model=load_model("Plant_disease_18-11-2020.h5")

plant_dict = {0: 'Apple___Apple_scab', 1: 'Apple___Black_rot', 2: 'Apple___Cedar_apple_rust', 3: 'Apple___healthy', 4: 'Blueberry___healthy',
 5: 'Cherry_(including_sour)___Powdery_mildew', 6: 'Cherry_(including_sour)___healthy', 7: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 8: 'Corn_(maize)___Common_rust_',
 9: 'Corn_(maize)___Northern_Leaf_Blight', 10: 'Corn_(maize)___healthy', 11: 'Grape___Black_rot', 12: 'Grape___Esca_(Black_Measles)',
 13: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 14: 'Grape___healthy', 15: 'Orange___Haunglongbing_(Citrus_greening)', 16: 'Peach___Bacterial_spot',
 17: 'Peach___healthy', 18: 'Pepper,_bell___Bacterial_spot', 19: 'Pepper,_bell___healthy', 20: 'Potato___Early_blight',
 21: 'Potato___Late_blight', 22: 'Potato___healthy', 23: 'Raspberry___healthy', 24: 'Soybean___healthy',
 25: 'Squash___Powdery_mildew', 26: 'Strawberry___Leaf_scorch', 27: 'Strawberry___healthy', 28: 'Tomato___Bacterial_spot',
 29: 'Tomato___Early_blight', 30: 'Tomato___Late_blight', 31: 'Tomato___Leaf_Mold', 32: 'Tomato___Septoria_leaf_spot',
 33: 'Tomato___Spider_mites Two-spotted_spider_mite', 34: 'Tomato___Target_Spot', 35: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 36: 'Tomato___Tomato_mosaic_virus', 37: 'Tomato___healthy'}

def model_predict(img_path, model):

    img = load_img(img_path,target_size=(224,224))
    img = img_to_array(img)/255.0
    img = img.reshape(1,224,224,3)

    pred = model.predict(img)
    pred = plant_dict[pred.argmax()]
    plant_name = pred.split('___')[0]
    plant_health = pred.split('___')[-1]

    return (plant_name,plant_health)

@app.route('/predict', methods=['POST','GET'])
def predict():
    
    if request.method == 'POST':
        
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        plant_name, plant_health=model_predict(file_path,model)

        return render_template('pred.html',plant_name=plant_name,plant_health=plant_health,file_name=str(f.filename))
	
@app.route('/upload/<filename>')
def upload_img(filename):
    return send_from_directory("uploads", filename)

@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)