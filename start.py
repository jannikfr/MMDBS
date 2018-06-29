import os
import re
from flask import Flask, render_template, request
from controller import Controller
import db_connection
from mmdbs_image import MMDBSImage

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/'

controller = Controller()


@app.route('/', methods=['POST', 'GET'])
def start():
    picanz = controller.get_number_of_mmdbs_images()
    return callHtmlPage('', '', '', '', picanz, None, None)


@app.route('/do_db_search', methods=['POST', 'GET'])
def do_db_search():
    if request.method == 'POST':
        # read variables out of form
        result = request.form
        queryobject = request.files['picture']
        feature = result['feature']
        seg = result['segmentation']
        distance_function = result['distance_function']
        eigenval = result['numberEigenvalues']

        # query picture amount
        picanz = controller.get_number_of_mmdbs_images()

        # save uploaded Image on server
        thePath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'uploadimage.jpg')
        queryobject.save(thePath)

        # build image object
        temp_image = MMDBSImage()
        temp_image.set_image(thePath, '')
        similiar_objects = controller.get_similar_objects(temp_image, feature, seg, distance_function)

        return callHtmlPage(feature, distance_function, seg, eigenval, picanz, queryobject, similiar_objects)


def callHtmlPage(feat, distance_function, seg, eigenanz, picanz, qo, so):
    return render_template('index.html', feat=feat, distance_function=distance_function, seg=seg, eigenanz=eigenanz, picanz=picanz, qo=qo, so=so)
