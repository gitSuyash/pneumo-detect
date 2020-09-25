

from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import h5py
import cv2
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='model.h5'

# Load your trained model
# model = h5py.File(MODEL_PATH,'r')
# model = pickle.load(open(MODEL_PATH, 'rb'))
model = load_model('model.h5')


def model_predict(img_path, model):
    # img = image.load_img(img_path, target_size=(224, 224))
    img_arr = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    resized_arr = cv2.resize(img_arr,(150,150))

    
    x = np.array(resized_arr)
    
    x=x/255
    
    x = x.reshape(-1,150,150,1)
 
    # x = preprocess_input(x)

    preds = model.predict(x)
    # preds=np.argmax(preds, axis=1)
    if preds<=0.5:
        preds="Pneumonia"
    elif preds>0.5:
        preds="Normal"
    
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
