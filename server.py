from flask import Flask, jsonify, request, current_app
from flask_cors import CORS
from flaskrun import flaskrun
from werkzeug import secure_filename

import urllib, datetime, os, sys

import tf_classifier as tf

app = Flask(__name__)
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # Limit of 16Mb

clf = tf.Tensorflow_ImagePredictor()

@app.route('/')
def return_index():
    return current_app.send_static_file('index.html')

@app.route('/status')
def return_app_status():

    payload = dict()
    payload["status"] = "Success"
    payload["timestamp"] = str(datetime.datetime.now())
    return jsonify(payload)

@app.route('/classify_image', methods = ['GET', 'POST'])
def clf_image():
   print(request)
   if request.method == 'POST':

      file = request.files['imagefile']

      filename = secure_filename(file.filename)
      ext = filename.split(".")[-1:][0].lower()

      rslt = clf.predict(file.read(), ext)
      return jsonify(rslt)

flaskrun(app)
