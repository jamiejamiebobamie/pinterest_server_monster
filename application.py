# import boto3
import datetime
from flask_restplus import Api, Resource, fields
from flask import Flask, jsonify, request, make_response, abort, render_template, redirect, url_for
import pickle

from werkzeug.datastructures import FileStorage

import pandas as pd
import numpy as np


import os, sys
from PIL import Image
import cv2


import numpy as np
from sklearn.externals import joblib
from keras.preprocessing.image import img_to_array

import requests

application = app = Flask(__name__)
api = Api(app, version='1.0', title='Concept Art Monster Identifier', description='Monster Identification Service')

ns = api.namespace('jamiejamiebobamie', description='Methods')

single_parser = api.parser()
single_parser.add_argument('img', location='files',
                           type=FileStorage, required=True, help= 'uploadedImage')

logreg_classifier_from_joblib = joblib.load('logreg_monster_classifier.pkl')

def resize(img):
    size = 200, 200
    with open(img, 'rb') as file:
        outfile = os.path.splitext(file.name)[0] + ".png"
        im = Image.open(file)
        im = im.resize(size)#, Image.ANTIALIAS)
        # return im
        im.save(outfile, "PNG")

# https://github.com/erykml/mario_vs_wario/blob/master/mario_vs_wario.ipynb
def img_to_1d_greyscale(img_path):
    # function for loading, resizing and converting an image into greyscale
    # used for logistic regression
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    return(pd.Series(img.flatten()))


@ns.route('/')
class IdentifyGender(Resource):
    """Identifies if a picture contains a 'monster'."""
    @api.doc(parser=single_parser, description='Submit a picture.')
    def post(self):
        """POST route."""

        args = single_parser.parse_args()
        image_file = args.img
        image_file.save('image.png')
        # img = Image.open('image.png')
        resize('image.png')
        img = img_to_1d_greyscale('image.png')

        # print(img)

        # image_read = img.resize((200, 200))
        # image_read = img_to_1d_greyscale(image_read)
        # image = img_to_array(image_read)
        # print(image.shape)
        # x = image.reshape(1, 200, 200, 1)
        # x = x/255
        # resize to 200 by 200
        # uploadedImage = resize(uploadedImage)



        # defining empty container
        X_test_sample = [[0]*200] * 200
        # print(len(img),len(X_test_sample), X_test_sample[-1][-1])


        count = 0
        x = []
        for i, row in enumerate(X_test_sample):
            for j, column in enumerate(row):
                x.append(img[count] / 255)
                # X_test_sample[i][j] = img[count] / 255
                count+=1

        # print(len(img),len(X_test_sample)*len(X_test_sample[0]), X_test_sample[-1][-1])

        r = logreg_classifier_from_joblib.predict([x])

        print(r)

        output = r[0]

        LOOKUP = {0:'not a monster', 1:'monster'}

        return {'Monster?': LOOKUP[output]}

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=5000)

#
# from keras.preprocessing.image import img_to_array
# from keras.models import load_model
# from flask_restplus import Api, Resource, fields
# from flask import Flask, request, jsonify
# import numpy as np
# from werkzeug.datastructures import FileStorage
# from PIL import Image
# from keras.models import model_from_json
# import tensorflow as tf
#
#
# app = Flask(__name__)
# api = Api(app, version='1.0', title='MNIST Classification', description='CNN for Mnist')
# ns = api.namespace('Make_School', description='Methods')
#
# single_parser = api.parser()
# single_parser.add_argument('file', location='files',
#                            type=FileStorage, required=True)
#
# logreg_classifier_from_joblib = joblib.load('logreg_classifier.pkl')
#
# # model = load_model('logreg_classifier.pkl')
# graph = tf.get_default_graph()
#
# # Model reconstruction from JSON file
# # with open('model_architecture.json', 'r') as f:
# #     model = model_from_json(f.read())
# #
# # # Load weights into the new model
# # model.load_weights('model_weights.h5')
#
#
# @ns.route('/prediction')
# class CNNPrediction(Resource):
#     """Uploads your data to the CNN"""
#     @api.doc(parser=single_parser, description='Upload an mnist image')
#     def post(self):
#         args = single_parser.parse_args()
#         image_file = args.file
#         image_file.save('milad.png')
#         img = Image.open('milad.png')
#         image_red = img.resize((200, 200))
#         image = img_to_array(image_red)
#         print(image.shape)
#         x = image.reshape(1, 200, 200, 1)
#         x = x/255
#         # This is not good, because this code implies that the model will be
#         # loaded each and every time a new request comes in.
#         # model = load_model('my_model.h5')
#         with graph.as_default():
#             out = model.predict(x)
#         print(out[0])
#         print(np.argmax(out[0]))
#         r = np.argmax(out[0])
#
#         return {'prediction': str(r)}
#
#
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8000)
