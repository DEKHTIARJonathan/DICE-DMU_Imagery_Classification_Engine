import os, sys, io, datetime, urllib

import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.append("inception_files")

from datasets import imagenet
from nets.inception_v1 import inception_v1, inception_v1_arg_scope, inception_v1_base
from preprocessing import inception_preprocessing

slim            = tf.contrib.slim

IMAGE_SIZE      = inception_v1.default_image_size

MODEL_REPO_URL  = "https://raw.githubusercontent.com/DEKHTIARJonathan/DICE-DMU_Imagery_Classification_Engine/web-api"
MODEL_PATH      = "inception_files/models"
MODEL_NAME      = "inception_v1.ckpt"

MODEL_FILEPATH  = MODEL_PATH + "/" + MODEL_NAME
MODEL_URL       = MODEL_REPO_URL + "/" + MODEL_PATH + "/" +MODEL_NAME

JPEG_EXT_LIST   = ["jpg", "jpeg"]

class Tensorflow_ImagePredictor():

    sess              = None
    
    img_plh           = tf.placeholder(tf.uint8, shape=[None, None, 3])

    names             = imagenet.create_readable_names_for_imagenet_labels()

    def __init__(self):
        print("Tensorflow_ImagePredictor: Model Checking Starting ...")

        if not os.path.isdir(MODEL_PATH):
            os.makedirs(MODEL_PATH)

        if not os.path.isfile(MODEL_FILEPATH):
            urllib.request.urlretrieve (MODEL_URL, MODEL_FILEPATH)

        print("Tensorflow_ImagePredictor: Model Checking Finished ...\n")

        ########################################################################

        print("Tensorflow_ImagePredictor: Initialisation Starting ...")

        self.processed_image   = inception_preprocessing.preprocess_image(self.img_plh, IMAGE_SIZE, IMAGE_SIZE, is_training=False)
        self.processed_images  = tf.expand_dims(self.processed_image, 0)

        #Config is necessary in order to prevent windows GPU compute failure
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        # Create the model, use the default arg scope to configure the batch norm parameters.
        with slim.arg_scope(inception_v1_arg_scope()):
            self.logits, self._ = inception_v1(self.processed_images, num_classes=1001, is_training=False)

        self.probabilities = tf.nn.softmax(self.logits)

        self.init_fn           = slim.assign_from_checkpoint_fn(
                                    MODEL_FILEPATH,
                                    slim.get_variables_to_restore()
                                 )

        self.init_fn(self.sess)

        print("Tensorflow_ImagePredictor: Initialisation Finished ...")

    def predict(self, image):

        probabilities           = self. sess.run([self.probabilities], feed_dict={self.img_plh: image})
        probabilities           = probabilities[0][0]
        
        sorted_inds             = [pair[0] for pair in sorted(enumerate(probabilities), key=lambda x: x[1], reverse=True)]
        
        payload                 = dict()
        payload["status"]       = "success"
        payload["timestamp"]    = str(datetime.datetime.now())

        payload["results"]      = list()

        for i in range(5):

            tmp                 = dict()

            index               = sorted_inds[i]

            tmp["position"]     = i + 1 # First position is Number 0 so we need to add 1!
            tmp["probability"]  = 100*probabilities[index]
            tmp["class_name"]   = self.names[index]

            payload["results"].append(tmp)

        return payload
