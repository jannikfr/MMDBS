import re
from operator import itemgetter
import numpy as np
import os
import db_connection
import distance_calculator
from random import randint
import matplotlib.pyplot as plt


class Controller(object):

    def __init__(self):
        self.conn = db_connection.connect()
        self.mmdbs_data = db_connection.get_all_images(self.conn)
        self.weight_dic = {}
        self.weight_dic['global_histogram'] = 1
        self.weight_dic['local_histogram'] = 1
        self.weight_dic['global_edge_histogram'] = 1
        self.weight_dic['global_hue_histogram'] = 1
        self.weight_dic['color_moments'] = 1
        self.weight_dic['central_circle_color_histogram'] = 1
        self.weight_dic['contours'] = 1


    def extract_all_features(self, mmdbs_image):
        mmdbs_image.global_histogram = self.extract_global_histogram_feature(mmdbs_image)
        mmdbs_image.local_histogram_2_2 = self.extract_local_histogram_feature(mmdbs_image, '2_2')
        mmdbs_image.local_histogram_3_3 = self.extract_local_histogram_feature(mmdbs_image, '3_3')
        mmdbs_image.local_histogram_4_4 = self.extract_local_histogram_feature(mmdbs_image, '4_4')
        mmdbs_image.global_edge_histogram = self.extract_global_edge_histogram_feature(mmdbs_image)
        mmdbs_image.global_hue_histogram = self.extract_global_hue_histogram_feature(mmdbs_image)
        mmdbs_image.color_moments = self.extract_color_moments_feature(mmdbs_image)
        mmdbs_image.central_circle_color_histogram = self.extract_central_circle_color_histogram_feature(mmdbs_image)
        mmdbs_image.contours = self.extract_contours_feature(mmdbs_image)

        return mmdbs_image

    def get_similar_objects(self, query_object, feature_list, seg, distance_function):
        """
        Extracts and sets the feature as attributes of the MMDBSImage object.
        :param feature: String identifier of the feature.
        """
        query_object_feature_dic = {}
        # extract features of query object corresponding the parameters
        for feature in feature_list:
            if feature == 'local_histogram':
                query_object_feature_dic[feature] = self.extract_local_histogram_feature(query_object, seg)
            
            elif feature == 'global_histogram':
                query_object_feature_dic[feature] = self.extract_global_histogram_feature(query_object)

            elif feature == 'global_edge_histogram':
                query_object_feature_dic[feature] = self.extract_global_edge_histogram_feature(query_object)

            elif feature == 'global_hue_histogram':
                query_object_feature_dic[feature] = self.extract_global_hue_histogram_feature(query_object)

            elif feature == 'color_moments':
                query_object_feature_dic[feature] = self.extract_color_moments_feature(query_object)

            elif feature == 'central_circle_color_histogram':
                query_object_feature_dic[feature] = self.extract_central_circle_color_histogram_feature(query_object)

            elif feature == 'contours':
                query_object_feature_dic[feature] = self.extract_contours_feature(query_object)

        # compute all distances for the selected feature
        similar_objects = self.get_all_distances(query_object_feature_dic, feature_list, seg, distance_function)
        # order list by distance
        similar_objects = sorted(similar_objects, key=itemgetter('distance'))
        return similar_objects

    def get_all_distances(self, query_object_feature_dic, feature_list, seg, distance_function):
        all_mmdbs_images = self.mmdbs_data
        similar_objects = []
        # loop over all images
        for mmdbs_image in all_mmdbs_images:
            # get distance between query object and mmdbs_image for this parameter and append it to the list
            similar_objects.append(
                self.get_distance(query_object_feature_dic, feature_list, seg, distance_function, mmdbs_image))

        return similar_objects

    def get_number_of_mmdbs_images(self):
        return db_connection.get_count_images(self.conn)

    def get_distance(self, query_object_feature_dic, feature_list, seg, distance_function, mmdbs_image):
        # read the selected feature from the mmdbs_image (database object)
        mmdbs_image_feature_dic = self.get_mmdbs_image_feature_dic(feature_list, seg, mmdbs_image)
        # build the mmdbs_image_distance dic with mmdbs_image:distance
        return self.get_mmdbs_image_distance_dictionary(mmdbs_image_feature_dic, query_object_feature_dic, distance_function,
                                                        mmdbs_image, feature_list)

    def get_mmdbs_image_feature_dic(self, feature_list, seg, mmdbs_image):
        # read the selected feature from the mmdbs_image
        mmdbs_image_feature_dic = {}
        for feature in feature_list:
            if feature == 'global_histogram':
                mmdbs_image_feature_dic[feature]=mmdbs_image.global_histogram

            elif feature == 'global_edge_histogram':
                mmdbs_image_feature_dic[feature] = mmdbs_image.global_edge_histogram

            elif feature == 'local_histogram':
                if seg == '2_2':
                    mmdbs_image_feature_dic[feature] = mmdbs_image.local_histogram_2_2

                elif seg == '3_3':
                    mmdbs_image_feature_dic[feature] = mmdbs_image.local_histogram_3_3

                elif seg == '4_4':
                    mmdbs_image_feature_dic[feature] = mmdbs_image.local_histogram_4_4

            elif feature == 'global_hue_histogram':
                mmdbs_image_feature_dic[feature] = mmdbs_image.global_hue_histogram

            elif feature == 'color_moments':
                mmdbs_image_feature_dic[feature] = mmdbs_image.color_moments

            elif feature == 'central_circle_color_histogram':
                mmdbs_image_feature_dic[feature] = mmdbs_image.central_circle_color_histogram

            elif feature == 'contours':
                mmdbs_image_feature_dic[feature] = mmdbs_image.contours
        return mmdbs_image_feature_dic


    def get_mmdbs_image_distance_dictionary(self, mmdbs_image_feature_dic, query_object_feature_dic, distance_function,
                                            mmdbs_image, feature):
        # initialize dic
        mmdbs_image_distance_dictonary = {}
        # set image as key
        mmdbs_image_distance_dictonary['mmdbs_image'] = mmdbs_image
        # get feature value for distance calculation
        mmdbs_image_feature_value_dic = self.get_value_for_distance_calculation(mmdbs_image_feature_dic, distance_function)
        query_object_feature_value_dic = self.get_value_for_distance_calculation(query_object_feature_dic, distance_function)

        # call distance function for calculation
        distance = distance_calculator.calculate_distance(mmdbs_image_feature_value_dic, query_object_feature_value_dic,
                                                          distance_function, self)
        # set distance as value
        mmdbs_image_distance_dictonary['distance'] = distance

        return mmdbs_image_distance_dictonary

    def get_value_for_distance_calculation(self, mmdbs_image_feature_dic, distance_function):
        # choose correct value for calculation dependent on feature...temporary all do the same
        mmdbs_image_feature_dic_new = mmdbs_image_feature_dic.copy()
        for key, value in mmdbs_image_feature_dic_new.items():
            if key != 'color_moments' and key != 'contours':
                mmdbs_image_feature_dic_new[key] = mmdbs_image_feature_dic_new[key]['cell_histograms'][0]['values']
        return mmdbs_image_feature_dic_new

    def extract_local_histogram_feature(self, mmdbs_image, seg):

        # extract local histogram feature corresponding to segmentation parameter
        if seg == '2_2':
            mmdbs_image.local_histogram_2_2 = mmdbs_image.extract_histograms(mmdbs_image.image, 2, 2, [8, 2, 4],
                                                                             False)
            return mmdbs_image.local_histogram_2_2

        elif seg == '3_3':
            mmdbs_image.local_histogram_3_3 = mmdbs_image.extract_histograms(mmdbs_image.image, 3, 3, [8, 2, 4],
                                                                             False)
            return mmdbs_image.local_histogram_3_3

        elif seg == '4_4':
            mmdbs_image.local_histogram_4_4 = mmdbs_image.extract_histograms(mmdbs_image.image, 4, 4, [8, 2, 4],
                                                                             False)
            return mmdbs_image.local_histogram_4_4

    def extract_global_histogram_feature(self, mmdbs_image):
        mmdbs_image.global_histogram = mmdbs_image.extract_histograms(mmdbs_image.image, 1, 1, [8, 2, 4], False)
        return mmdbs_image.global_histogram

    def extract_global_edge_histogram_feature(self, mmdbs_image):
        mmdbs_image.sobel_edges = self.extract_sobel_edges(mmdbs_image)
        min_edge_value = np.min(mmdbs_image.sobel_edges)
        max_edge_value = np.max(mmdbs_image.sobel_edges)
        mmdbs_image.global_edge_histogram = mmdbs_image.extract_histograms_one_channel(mmdbs_image.sobel_edges,
                                                                                       1, 1,
                                                                                       64,
                                                                                       False,
                                                                                       min_edge_value,
                                                                                       max_edge_value)
        return mmdbs_image.global_edge_histogram

    def extract_global_hue_histogram_feature(self, mmdbs_image):
        h_image = mmdbs_image.image[:, :, 0]
        min_h_value = np.min(h_image)
        max_h_value = np.max(h_image)
        mmdbs_image.global_hue_histogram = mmdbs_image.extract_histograms_one_channel(h_image, 1, 1, 64, False,
                                                                                      min_h_value, max_h_value)
        return mmdbs_image.global_hue_histogram

    def extract_color_moments_feature(self, mmdbs_image):
        mmdbs_image.color_moments = mmdbs_image.extract_color_moments(mmdbs_image.image)
        return mmdbs_image.color_moments

    def extract_central_circle_color_histogram_feature(self, mmdbs_image):
        # Get modified circle image and apply histogram on it
        central_circle = mmdbs_image.get_central_circle(mmdbs_image.image.copy())
        mmdbs_image.central_circle_color_histogram = mmdbs_image.extract_histograms(central_circle, 1, 1, [8, 2, 4],
                                                                                    False)
        return mmdbs_image.central_circle_color_histogram

    def extract_contours_feature(self, mmdbs_image):
        # Extraction of contours. Needs sobel edge data
        if mmdbs_image.sobel_edges is None:
            mmdbs_image.sobel_edges = self.extract_sobel_edges(mmdbs_image)
        mmdbs_image.contours = mmdbs_image.extract_contours(mmdbs_image.image, mmdbs_image.sobel_edges)
        return mmdbs_image.contours

    def extract_sobel_edges(self, mmdbs_image):
        mmdbs_image.sobel_edges = mmdbs_image.extract_sobel_edges(mmdbs_image.image)
        return mmdbs_image.sobel_edges

    @staticmethod
    def plot_precision_recall_curve(similar_objects, correct_classification, number_of_results):
        """
        Plots and saves a precision recall curve based on on the k best results.
        The filename is precision_recall.png'.
        :param similar_objects: Result set
        :param correct_classification: Correct classication
        :param number_of_results: The number of results, which should be considered for the calculation
        """

        # Initialize parameters
        precision = []
        recall = []
        accumulated_correct_objects = [0]
        overall_correct_objects = 0

        # Loop over result object
        for result_object in similar_objects[:number_of_results]:

            # Get classification of current result object
            result_object_classification = result_object['mmdbs_image'].classification

            # Current result object is classified correctly
            if result_object_classification == correct_classification:
                # Increase overall counter for correct result objects
                overall_correct_objects = overall_correct_objects + 1
                # Increase last value by 1 and add it to the timeline
                accumulated_correct_objects.append(accumulated_correct_objects[-1]+1)
            else:
                # Add the same as last value to the timeline
                accumulated_correct_objects.append(accumulated_correct_objects[-1])

        # Remove first initial value
        accumulated_correct_objects = accumulated_correct_objects[1:number_of_results]

        # Calculate precision and recall for the number_of_results
        for i, value in enumerate(accumulated_correct_objects):
            k = i + 1
            precision.append(value/k)
            recall.append(value/overall_correct_objects)

        # Plot results
        fig, ax = plt.subplots()
        ax.set_title(r'Precision-Recall-Curve of the '+str(number_of_results)+' best results.')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        fig.tight_layout()

        # Line chart with markers at each data point
        plt.plot(recall, precision, marker='o', markersize=5)
        plt.axis([0, 1.1, 0, 1.1])

        # create unique filename
        path = 'static/precision' + str(randint(0, 100000)) + '.png'

        # Export plot to file
        plt.savefig(path)

        return path

    @staticmethod
    def normalize_distances(similar_objects, number_of_results):
        """
        Normalize the distances linearly.
        :param similar_objects: Result set
        :param number_of_results: The number of results, which should be considered for the calculation
        :return: The Result set array enriched by the attribute 'normalized_distance'
        """

        # Get lower and upper boundary
        min_distance = similar_objects[0]['distance']
        max_distance = similar_objects[number_of_results-1]['distance']

        # Loop over result set
        for similar_object in similar_objects[:number_of_results]:

            # Calculate normalized distance
            distance = similar_object['distance']
            normalized_distance = (distance - min_distance)/(max_distance - min_distance)
            similar_object['normalized_distance'] = format(normalized_distance, '.2f')

        return similar_objects

    @staticmethod
    def get_upload_image_path(queryobject):
        upload_image_path = 'static/uploadimage' + str(randint(0, 100000)) + '.jpg'
        save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), upload_image_path)
        queryobject.save(save_path)
        return upload_image_path



    # @staticmethod
    # def delete_images_on_server():
        # for f in os.listdir(os.path.join(os.path.dirname(os.path.realpath(__file__)),'static')):
            # if re.search(pattern, f):
            #   os.remove(os.path.join(dir, f))
