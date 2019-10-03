from flask_restplus import Api, Resource, fields
from flask import Flask, jsonify, request, make_response, abort, render_template, redirect, url_for
from werkzeug.datastructures import FileStorage

import pandas as pd

import os, sys
from PIL import Image
import cv2

from sklearn.externals import joblib

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
        im = im.resize(size)
        im.save(outfile, "PNG")

def img_to_1d_greyscale(img_path):
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
        resize('image.png')
        img = img_to_1d_greyscale('image.png')

        X_test_sample = [[0]*200] * 200

        count = 0
        x = []
        for i, row in enumerate(X_test_sample):
            for j, column in enumerate(row):
                x.append(img[count] / 255)
                count+=1

        r = logreg_classifier_from_joblib.predict([x])

        print(r)

        output = r[0]

        LOOKUP = {0:'not a monster', 1:'monster'}

        return {'Monster?': LOOKUP[output]}

if __name__ == '__main__':
    app.run(debug=True)
    # app.run(host='0.0.0.0', port=5000)
