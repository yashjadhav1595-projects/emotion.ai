from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

app = Flask(__name__, static_folder="../static", template_folder="../templates")
model = load_model(os.path.join(os.path.dirname(__file__), '../emotion_model.h5'))
emotions = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '../static/uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files or request.files['image'].filename == '':
            return render_template('index.html', error='No file selected. Please upload an image.')
        file = request.files['image']
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (48,48))
        img = img / 255.0
        img = np.expand_dims(img, axis=(0,-1))

        prediction = model.predict(img)
        emotion = emotions[np.argmax(prediction)]

        image_path = '/static/uploads/' + file.filename
        return render_template('result.html', image_path=image_path, emotion=emotion)

    return render_template('index.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.static_folder, filename)

def handler(environ, start_response):
    from werkzeug.middleware.dispatcher import DispatcherMiddleware
    return DispatcherMiddleware(app)(environ, start_response)
