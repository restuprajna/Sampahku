import logging
from typing import Union
import os
import json
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
from google.cloud import storage

app = Flask(__name__)

# Configure this environment variable via app.yaml
CLOUD_STORAGE_BUCKET = 'sampaku-data-image'

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


@app.route('/')
def index() -> str:
    return """
<form method="POST" action="/upload" enctype="multipart/form-data">
    <input type="file" name="file">
    <input type="submit">
</form>
"""


@app.route('/upload', methods=['GET', 'POST'])
def upload() -> str:
    """Process the uploaded file and upload it to Google Cloud Storage."""
    uploaded_file = request.files.get('image')

    if not uploaded_file:
        return 'No file uploaded.', 400

    # Create a Cloud Storage client.
    gcs = storage.Client()

    # Get the bucket that the file will be uploaded to.
    bucket = gcs.get_bucket(CLOUD_STORAGE_BUCKET)

    # Create a new blob and upload the file's content.
    blob = bucket.blob(uploaded_file.filename)
    blob.make_public()
    filepath = os.path.join(bucket, blob)
    uploaded_file.save(filepath)
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

        prediction = result
        #return position of max
        MaxPosition=np.argmax(prediction)  
        prediction_label=classes[MaxPosition]
                        
        json_object = json.dumps(prediction_label, indent = 4)
    return str(json_object)              

    # return str(data)

    # blob.upload_from_string(
    #     uploaded_file.read(),
    #     content_type=uploaded_file.content_type
    # )

    # # Then do other things...
    # blob = bucket.get_blob('remote/path/to/file.txt')

    # # Make the blob public. This is not necessary if the
    # # entire bucket is public.
    # # See https://cloud.google.com/storage/docs/access-control/making-data-public.
    # blob.make_public()

    # # The public URL can be used to directly access the uploaded file via HTTP.
    # return blob.public_url


@app.errorhandler(500)
def server_error(e: Union[Exception, int]) -> str:
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)