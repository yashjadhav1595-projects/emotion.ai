from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

app = Flask(__name__)
model = load_model('emotion_model.h5')
emotions = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        # Preprocess image
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (48,48))
        img = img / 255.0
        img = np.expand_dims(img, axis=(0,-1))

        # Predict emotion
        prediction = model.predict(img)
        emotion = emotions[np.argmax(prediction)]

        return render_template('result.html', image_path=path, emotion=emotion)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

