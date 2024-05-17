from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
model = load_model(r'C:\Users\abdul\Downloads\project\model.h5')

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            image_path = 'temp_image.jpg'
            image_file.save(image_path)
            img_array = preprocess_image(image_path)
            prediction = model.predict(img_array)
            if prediction[0][0] > 0.5:
                result = 'Pneumonia'
            else:
                result = 'Normal'
            return render_template('index.html', prediction=result)
    return render_template('index.html', prediction='Error')

if __name__ == '__main__':
    app.run(debug=True)
