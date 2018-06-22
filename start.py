import numpy as np
import os

import cv2
from flask import Flask, render_template, request
import db_connection
import feature_extractor
from image import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/'


@app.route('/', methods=['POST', 'GET'])
def start():
    queryobject = None
    feat = None
    sim = None
    seg = None
    eigenval = None
    picanz = db_connection.get_count_images()
    return callHtmlPage(feat, sim, seg, eigenval, picanz, queryobject)


@app.route('/do_db_search', methods=['POST', 'GET'])
def do_db_search():
    if request.method == 'POST':
        # read variables out of form
        result = request.form
        queryobject = request.files['picture']
        feat = result['feature']
        seg = result['segmentation']
        sim = result['similarity']
        eigenval = result['numberEigenvalues']

        # query picture amount
        picanz = db_connection.get_count_images()

        # save uploaded Image on server
        thePath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'uploadimage.jpg')
        queryobject.save(thePath)

        # build image object
        temp_image = Image()
        temp_image.buildAttributes(thePath, '')

        # extract local histogram
        temp_image.local_histogram = feature_extractor.extract_histograms(temp_image.image, 1, 2, [8, 2, 4],
                                                                          False)
        # extract global histogram
        temp_image.global_histogram = feature_extractor.extract_histograms(temp_image.image, 1, 1, [8, 2, 4],
                                                                           False)
        # extract sobel edge
        temp_image.sobel_edge_detection = feature_extractor.sobel_edge_detection(temp_image.image)

        # extract global edge histogram
        temp_image.global_edge_histogram = feature_extractor.extract_histograms_greyscale(
            temp_image.sobel_edge_detection, 1, 1, 64, False, np.min(temp_image.sobel_edge_detection),
            np.max(temp_image.sobel_edge_detection))

        return callHtmlPage(feat, sim, seg, eigenval, picanz, queryobject)


def callHtmlPage(feat, sim, seg, eigenanz, picanz, qo):
    return render_template('index.html', feat=feat, sim=sim, seg=seg, eigenanz=eigenanz, picanz=picanz, qo=qo)
