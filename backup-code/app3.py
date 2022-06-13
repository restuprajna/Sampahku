import os
import io
import json
import flask
import werkzeug
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
# folder to storage, can be changed into cloud storage?
app.config["UPLOAD_FOLDER"] = "Upload"

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
    # img = tf.keras.applications.imagenet_utils.preprocess_input(img)
    # return the processed image
    return img

classes=['cardboard', 'compost', 'glass', 'metal', 'paper', 'plastic']


@app.route('/', methods=["GET", "POST"])
def upload_file():
    data = {"success": False}
    if request.method == "POST":
        if request.files.get("image"):
            file = flask.request.files["image"]
        #    filename = file.filename
            filename = werkzeug.utils.secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
           # Load the saved image using Keras.
            # Resize it to the Xception format of 299x299 pixels.
            image_size = (224, 224)
            im = keras.preprocessing.image.load_img(filepath,
                                                    target_size=image_size)

            # Preprocess the image and prepare it for classification.
            image = prepare_image(im)

            global graph
            with graph.as_default():
                model = keras.models.load_model("ml-model/my_model1.h5", compile = False)
                graph = tf.compat.v1.Session().graph
                result = model.predict(image)
                print(result)
                # results = decode_predictions(preds)
                data["predictions"] = []

                prediction = result
                #return position of max
                MaxPosition=np.argmax(prediction)  
                prediction_label=classes[MaxPosition]
                    
                # labelName = names[int(label)]
                r = {"labelName": prediction_label}
                data["predictions"].append(r)
                # indicate that the request was a success
                data["success"] = True
                kategori = ""
                if(prediction_label == 'compost'):
                    kategori = "Organic"
                else:
                    kategori = "inorganic"
                json_object = json.dumps(kategori, indent = 4)
                return str(json_object)              

        return str(data)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)