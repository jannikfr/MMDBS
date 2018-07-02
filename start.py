import re
from operator import itemgetter

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
    return callHtmlPage('', '', '',  picanz, None, None, feature_methods, distance_functions, segments, amount_results, None, None, controller)


@app.route('/do_db_search', methods=['POST', 'GET'])
def do_db_search():
    if request.method == 'POST':
        feature_methods_list = get_feature_methods()
        distance_functions_list = get_distance_functions()
        segments_list = get_segments()

        # read variables out of form
        result = request.form
        queryobject = request.files['picture']
        feature_list = request.form.getlist('feature')
        seg = result['segmentation']
        distance_function = result['distance_function']
        amount_results = int(result['amount_results'])
        for feature_method in feature_list:
            my_result = float(result[feature_method + '/'])
            controller.weight_dic[feature_method] = my_result

        # query picture amount
        picanz = controller.get_number_of_mmdbs_images()

        # delete images from last run
        # controller.delete_images_on_server()

        # save uploaded Image on server
        upload_image_path = controller.get_upload_image_path(queryobject)

        # build image object
        temp_image = MMDBSImage()
        temp_image.set_image(upload_image_path, '')

        # calculate distances to all images in database
        similar_objects = controller.get_similar_objects(temp_image, feature_list, seg, distance_function)

        # prepare upload path for HTML
        upload_image_path = '/' + upload_image_path



        # normalize distances
        similar_objects = controller.normalize_sub_distances(similar_objects)

        similar_objects = sorted(similar_objects, key=itemgetter('distance'))

        similar_objects = controller.normalize_distances(similar_objects, amount_results)
        # do precision recall diagramm

        precision_recall_path = '/' + controller.plot_precision_recall_curve(similar_objects, similar_objects[0][
            'mmdbs_image'].classification, amount_results)

        return callHtmlPage(feature_list, distance_function, seg, picanz, queryobject, similar_objects, feature_methods_list, distance_functions_list, segments_list, amount_results, precision_recall_path, upload_image_path, controller)


def callHtmlPage(feat, selected_distance_function, seg, picanz, qo, so, fm, df, segs, ar, prp, up, cont):
    return render_template('index.html', feat=feat, sdf=selected_distance_function, seg=seg, picanz=picanz, qo=qo, so=so, fm=fm, df=df, segs=segs, ar=ar , prp=prp, up=up, controller=cont)

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
    functions.append(['cosine_distance','Cosine Distance'])
    functions.append(['weighted_euclidean_distance','Weighted Euclidean Distance'])
    return functions


def get_segments():
    segments =[]
    segments.append(['2_2','2x2'])
    segments.append(['3_3','3x3'])
    segments.append(['4_4','4x4'])
    return segments
