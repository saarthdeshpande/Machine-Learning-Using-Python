from flask import Flask,jsonify,request
import cv2
import numpy as np
from urllib.request import urlretrieve
import keras
import tensorflow as tf
import os

config = tf.ConfigProto(
    device_count={'GPU': 1},
    intra_op_parallelism_threads=1,
    allow_soft_placement=True
)

config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6

session = tf.Session(config=config)

keras.backend.set_session(session)

model = keras.models.load_model('best_model.hdf5')

app = Flask(__name__)

@app.route('/predict',methods=['POST'])
def predict():
    '''
    :param image_location: location in storage of image to be analysed
    :return: boolean value to predict if rotten or not
    '''
    response = request.get_json()
    image_url = response["url"]
    try:
        with session.as_default():
            with session.graph.as_default():
                urlretrieve(image_url,'image.png')
                image = cv2.imread('image.png',cv2.IMREAD_UNCHANGED)
                os.remove('image.png')
                image = cv2.resize(image, (200, 200))
                image = np.expand_dims(image, axis=0)
                return jsonify({"rotten": str(int(model.predict(image)[0][0]))})
    except Exception as e:
        return jsonify({"Error":str(e)})

app.run(port=8080, host='0.0.0.0')
