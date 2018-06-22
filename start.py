import os
from flask import Flask, render_template, request
import db_connection
from mmdbs_image import MMDBSImage

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
        temp_image = MMDBSImage()
        temp_image.set_image(thePath, '')
        temp_image.extract_features(feat)


        return callHtmlPage(feat, sim, seg, eigenval, picanz, queryobject)


def callHtmlPage(feat, sim, seg, eigenanz, picanz, qo):
    return render_template('index.html', feat=feat, sim=sim, seg=seg, eigenanz=eigenanz, picanz=picanz, qo=qo)
