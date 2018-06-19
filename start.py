import os

import cv2
from flask import Flask, render_template, request
import db_connection

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
        result = request.form
        queryobject = request.files['picture']
        feat = result['feature']
        #save uploaded Image
        thePath2 = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'uploadimage.jpg')
        queryobject.save(thePath2)
        #read uploaded image with cv2

        seg = result['segmentation']
        sim = result['similarity']
        eigenval = result['numberEigenvalues']
        # extract feature
        picanz = db_connection.get_count_images()
        return callHtmlPage(feat, sim, seg, eigenval, picanz, queryobject)


def callHtmlPage(feat, sim, seg, eigenanz, picanz, qo):
    return render_template('index.html', feat=feat, sim=sim, seg=seg, eigenanz=eigenanz, picanz=picanz, qo=qo)
