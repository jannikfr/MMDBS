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
    queryobject = None
    feat = None
    sim = None
    seg = None
    classification = None
    eigenval = None
    similiar_objects = None
    picanz = controller.get_number_of_mmdbs_images()
    return callHtmlPage(feat, sim, seg, eigenval, picanz, queryobject, similiar_objects, classification)


@app.route('/do_db_search', methods=['POST', 'GET'])
def do_db_search():
    if request.method == 'POST':
        # read variables out of form
        result = request.form
        queryobject = request.files['picture']
        feature = result['feature']
        seg = result['segmentation']
        sim = result['similarity']
        eigenval = result['numberEigenvalues']

        # query picture amount
        picanz = controller.get_number_of_mmdbs_images()

        # save uploaded Image on server
        thePath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'uploadimage.jpg')
        queryobject.save(thePath)

        # build image object
        temp_image = MMDBSImage()
        temp_image.set_image(thePath, '')
        similiar_objects = controller.get_similar_objects(temp_image, feature, seg)
        for similiar_object in similiar_objects:
            path = similiar_object['mmdbs_image'].path
            pattern = re.compile('/static.+')
            similiar_object['mmdbs_image'].path = pattern.findall(path)[0]
        classification='accordion'
        return callHtmlPage(feature, sim, seg, eigenval, picanz, queryobject, similiar_objects, classification)


def callHtmlPage(feat, sim, seg, eigenanz, picanz, qo, so, cl):
    return render_template('index.html', feat=feat, sim=sim, seg=seg, eigenanz=eigenanz, picanz=picanz, qo=qo, so=so, cl=cl)
