import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import io
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from PIL import Image
import numpy as np
import tensorflow as tf

from flask import Flask, request, jsonify

model = tf.keras.models.load_model('image_class_beta.h5')


# def transform_image(pillow_image):
#     data = np.asarray(pillow_image)
#     data = data / 255.0
#     data = data[np.newaxis, ..., np.newaxis]
#     # --> [1, x, y, 1]
#     data = tf.image.resize(data, [64, 64])
#     return data
#masih mentah


# def predict(x):
#     predictions = model(x)
#     predictions = tf.nn.softmax(predictions)
#     pred0 = predictions[0]
#     label0 = np.argmax(pred0)
#     return label0

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({"error": "no file"})

        try:
            image_bytes = file.read()
            pillow_img = Image.open(io.BytesIO(image_bytes)).convert('L')
            #tensor = transform_image(pillow_img)
            test_image = np.asarray(pillow_img)
            test_image = np.expand_dims(test_image, axis = 0)
            result = model.predict(test_image)
            if result[0][0] == 1:
                prediction = 'Recyclable'
            else:
                prediction = 'Organic'
            return jsonify(prediction)
        except Exception as e:
            return jsonify({"error": str(e)})

    return "OK"


if __name__ == "__main__":
    app.run(debug=True)