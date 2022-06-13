import os
import io
import numpy as np
import json
import tensorflow as tf
import flask
import werkzeug
from PIL import Image

import keras
from keras.preprocessing import image
#from keras.preprocessing.image import img_to_array
from keras.applications.xception import (
    Xception, preprocess_input)
#from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow import keras
from keras import backend as K

from flask_ngrok import run_with_ngrok
from flask import Flask, request, redirect, url_for, jsonify, Response

app = Flask(__name__)
# run_with_ngrok(app)
# folder to storage, can be changed into cloud storage?
app.config["UPLOAD_FOLDER"] = "Upload"

model = None
graph = None


def model_load():
    global model
    global graph
    model = tf.keras.models.load_model("image_class_beta.h5")
    graph = tf.compat.v1.Session().graph


model_load()


def prepare_image(img):
    img = tf.keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    # img = preprocess_input(img)
    # return the processed image
    return img

names = ["Organic","Recyclable"]

#@app.route("/")
#def hello():
#    return "Hello, World!"

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
            image_size = (64, 64)
            im = keras.preprocessing.image.load_img(filepath,
                                                    target_size=image_size)

            # Preprocess the image and prepare it for classification.
            image = prepare_image(im)

            global graph
            with graph.as_default():
                model = tf.keras.models.load_model("image_class_beta.h5")
                graph = tf.compat.v1.Session().graph
                preds = model.predict(image)
                print(preds)
                # results = decode_predictions(preds)
                data["predictions"] = []

                # for (imagenetID, label, prob) in results[0]:
                #    r = {"label": label, "probability": float(prob)}
                #    data["predictions"].append(r)

                prob = preds
                if preds[0][0] == 1:
                    labelName = names[1]
                else:
                    labelName = names[0] 
                    
                # labelName = names[int(label)]
                r = {"labelName": labelName, "prob": float(prob)}
                data["predictions"].append(r)

                # indicate that the request was a success
                data["success"] = True
                json_object = json.dumps(r, indent = 4)
                
                return str(json_object)

        return str(data)

if __name__ == "__main__":
    app.run()