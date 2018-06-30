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
    feature_methods = get_feature_methods()
    distance_functions = get_distance_functions()
    segments = get_segments()
    amount_results = 20
    return callHtmlPage('', '', '', '', picanz, None, None, feature_methods, distance_functions, segments, amount_results)


@app.route('/do_db_search', methods=['POST', 'GET'])
def do_db_search():
    if request.method == 'POST':
        feature_methods = get_feature_methods()
        distance_functions = get_distance_functions()
        segments = get_segments()

        # read variables out of form
        result = request.form
        queryobject = request.files['picture']
        feature = result['feature']
        seg = result['segmentation']
        distance_function = result['distance_function']
        eigenval = result['numberEigenvalues']
        amount_results = int(result['amount_results'])

        # query picture amount
        picanz = controller.get_number_of_mmdbs_images()

        # save uploaded Image on server
        thePath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'uploadimage.jpg')
        queryobject.save(thePath)

        # build image object
        temp_image = MMDBSImage()
        temp_image.set_image(thePath, '')
        similiar_objects = controller.get_similar_objects(temp_image, feature, seg, distance_function)

        return callHtmlPage(feature, distance_function, seg, eigenval, picanz, queryobject, similiar_objects, feature_methods, distance_functions, segments, amount_results)


def callHtmlPage(feat, selected_distance_function, seg, eigenanz, picanz, qo, so, fm, df, segs, ar):
    return render_template('index.html', feat=feat, sdf=selected_distance_function, seg=seg, eigenanz=eigenanz, picanz=picanz, qo=qo, so=so, fm=fm, df=df, segs=segs, ar=ar)

def get_feature_methods():
    methods =[]
    methods.append(['global_histogram','Global Color Histogram'])
    methods.append(['local_histogram','Local Color Histograms'])
    methods.append(['global_edge_histogram','Global Edge Histogram'])
    methods.append(['global_hue_histogram','Global Hue Histogram'])
    methods.append(['color_moments','Color Moments'])
    methods.append(['central_circle_color_histogram', 'Central Circle Color Histogram'])
    methods.append(['contours', 'Contours'])
    return methods

def get_distance_functions():
    functions =[]
    functions.append(['euclidean_distance','Euclidean Distance'])
    functions.append(['quadratic_form_distance','Quadratic Form Distance'])
    functions.append(['last_distance','Last Distance'])
    return functions


def get_segments():
    segments =[]
    segments.append(['2_2','2x2'])
    segments.append(['3_3','3x3'])
    segments.append(['4_4','4x4'])
    return segments
