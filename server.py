from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flaskrun import flaskrun
from werkzeug import secure_filename

import datetime

import tf_classifier as tf

app = Flask(__name__)
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 # Limit of 16Mb

clf = tf.Tensorflow_ImagePredictor()

@app.route('/')
def return_index():
    return app.send_static_file('index.html')

@app.route('/css/<path:path>')
def send_css(path):
    return app.send_static_file('css' + "/" + path)

@app.route('/images/<path:path>')
def send_img(path):
    return app.send_static_file('images' + "/" + path)

@app.route('/status')
def return_app_status():

    payload = dict()
    payload["status"] = "Success"
    payload["timestamp"] = str(datetime.datetime.now())
    return jsonify(payload)

@app.route('/classify_image', methods = ['GET', 'POST'])
def clf_image():

    if request.method == 'POST':

        file = request.files['imagefile']

        filename = secure_filename(file.filename)
        ext = filename.split(".")[-1:][0].lower()

        rslt = clf.predict(file.read(), ext)
        return jsonify(rslt)

    else:
        return "Only POST Requests are accepted"

if __name__ == "__main__":
    flaskrun(app)
