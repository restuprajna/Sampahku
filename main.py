import io
#import numpy for number array handling and represent rgb image pixel values
import numpy as np

#import tensorflow to use any tools needed for deep learning
import tensorflow as tf

#import keras api needed to implement deep learning techiques
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, BatchNormalization, Conv2D, MaxPool2D, GlobalAveragePooling2D, Dropout
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
from keras.preprocessing.image import ImageDataGenerator

from PIL import Image
from keras.preprocessing import image
from keras import backend as K

from flask_ngrok import run_with_ngrok
from flask import Flask, request, redirect, url_for, jsonify, Response


app = Flask(__name__)
run_with_ngrok(app)

model = None
graph = None


def model_load():
    global model
    global graph
    model = tf.keras.models.load_model("ml-model/my_model1.h5", compile = False)
    graph = tf.compat.v1.Session().graph


model_load()


def prepare_image(img):
    img = tf.keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img
#class for classification
classes=['cardboard', 'compost', 'glass', 'metal', 'paper', 'plastic']


@app.route('/', methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files.get('image')
        if file is None or file.filename == "":
            #return the non image error
            return jsonify({"error": "no image uploaded"})

        try:
            read_image = file.read()
            #save uploaded image in memory (still need improvment) can it be saved in cloud storage?
            save_to_memory = io.BytesIO(read_image)
            image_size = (224, 224)
            #preprocessing the uploaded image
            img = keras.preprocessing.image.load_img(save_to_memory,
                                                    target_size=image_size)
            image = prepare_image(img)
            
            global graph
            with graph.as_default():
                model = keras.models.load_model("ml-model/my_model1.h5", compile = False)
                graph = tf.compat.v1.Session().graph
                # Predict the image
                result = model.predict(image)
                #show the prediction result (only check)
                print(result)
                
                prediction = result
                # look the biggest value location (index number)
                max_index=np.argmax(prediction) 
                # see the class name by index location 
                prediction_label=classes[max_index]
                #return json format of prediction label
            return jsonify(prediction_label)
        except Exception as e:
            #return the error
            return jsonify({"error": str(e)})

    return "OK"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)