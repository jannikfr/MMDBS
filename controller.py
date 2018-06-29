import math

import cv2
from operator import itemgetter
import numpy as np
import db_connection
import distance_calculator


class Controller(object):

    def __init__(self):
        self.conn = db_connection.connect()
        self.mmdbs_data = db_connection.get_all_images(self.conn)

    @staticmethod
    def extract_all_features(mmdbs_image):
        mmdbs_image.global_histogram = mmdbs_image.extract_histograms(mmdbs_image.image, 1, 1, [8, 2, 4], False)
        mmdbs_image.local_histogram_2_2 = mmdbs_image.extract_histograms(mmdbs_image.image, 2, 2, [8, 2, 4], False)
        mmdbs_image.local_histogram_3_3 = mmdbs_image.extract_histograms(mmdbs_image.image, 3, 3, [8, 2, 4], False)
        mmdbs_image.local_histogram_4_4 = mmdbs_image.extract_histograms(mmdbs_image.image, 4, 4, [8, 2, 4], False)
        mmdbs_image.sobel_edges = mmdbs_image.extract_sobel_edges(mmdbs_image.image)
        min_edge_value = np.min(mmdbs_image.sobel_edges)
        max_edge_value = np.max(mmdbs_image.sobel_edges)
        mmdbs_image.global_edge_histogram = mmdbs_image.extract_histograms_one_channel(mmdbs_image.sobel_edges, 1, 1,
                                                                                       64,
                                                                                       False,
                                                                                       min_edge_value, max_edge_value)

        h_image = mmdbs_image.image[:, :, 0]
        min_h_value = np.min(h_image)
        max_h_value = np.max(h_image)
        mmdbs_image.global_hue_histogram = mmdbs_image.extract_histograms_one_channel(h_image, 1, 1, 64, False,
                                                                                      min_h_value, max_h_value)

        # Color moments
        mmdbs_image.color_moments = mmdbs_image.extract_color_moments(mmdbs_image.image)

        # Get modified circle image and apply histogram on it
        central_circle = mmdbs_image.get_central_circle(mmdbs_image.image.copy())
        mmdbs_image.central_circle_color_histogram = mmdbs_image.extract_histograms(central_circle, 1, 1, [8, 2, 4], False)

        # Extraction of contours. Needs sobel edge data
        mmdbs_image.contours = mmdbs_image.extract_contours(mmdbs_image.image, mmdbs_image.sobel_edges)

        return mmdbs_image


    def get_similar_objects(self, query_object, feature, seg, distance_function):
        """
        Extracts and sets the feature as attributes of the MMDBSImage object.
        :param feature: String identifier of the feature.
        """
        # extract features of query object corresponding the parameters
        if feature == 'local_histogram':
            query_object_feature = self.extract_local_histogram_feature(query_object, seg)

        elif feature == 'global_histogram':
            query_object_feature = self.extract_global_histogram_feature(query_object)

        elif feature == 'global_edge_histogram':
            query_object_feature = self.extract_global_edge_histogram_feature(query_object)
        # compute all distances for the selected feature
        similar_objects = self.get_all_distances(query_object_feature, feature, seg, distance_function)
        # order list by distance
        similar_objects = sorted(similar_objects, key=itemgetter('distance'))
        return similar_objects

    def get_all_distances(self, query_object_feature, feature, seg, distance_function):

        all_mmdbs_images = self.mmdbs_data
        similar_objects = []
        # loop over all images
        for mmdbs_image in all_mmdbs_images:
            # get distance between query object and mmdbs_image for this parameter and append it to the list
            similar_objects.append(self.get_distance(query_object_feature, feature, seg, distance_function, mmdbs_image))

        return similar_objects

    def get_number_of_mmdbs_images(self):
        return db_connection.get_count_images(self.conn)

    def get_distance(self, query_object_feature, feature, seg, distance_function, mmdbs_image):
        # read the selected feature from the mmdbs_image (database object)
        mmdbs_image_feature = self.get_mmdbs_image_feature(feature, seg, mmdbs_image)
        # build the mmdbs_image_distance dic with mmdbs_image:distance
        return self.get_mmdbs_image_distance_dictonary(mmdbs_image_feature, query_object_feature, distance_function,
                                                    mmdbs_image)

    def get_mmdbs_image_feature(self, feature, seg, mmdbs_image):
        # read the selected feature from the mmdbs_image
        if feature == 'global_histogram':
            return mmdbs_image.global_histogram['cell_histograms'][0]['values']

        elif feature == 'global_edge_histogram':
            return mmdbs_image.global_edge_histogram['cell_histograms'][0]['values']

        elif feature == 'local_histogram':
            if seg == '2_2':
                return mmdbs_image.local_histogram_2_2['cell_histograms'][0]['values']

            elif seg == '3_3':
                return mmdbs_image.local_histogram_3_3['cell_histograms'][0]['values']

            elif seg == '4_4':
                return mmdbs_image.local_histogram_4_4['cell_histograms'][0]['values']

    def get_mmdbs_image_distance_dictonary(self, mmdbs_image_feature, query_object_feature, distance_function,
                                           mmdbs_image):
        # initialize dic
        mmdbs_image_distance_dictonary = {}
        # set image as key
        mmdbs_image_distance_dictonary['mmdbs_image'] = mmdbs_image
        # call distance function for calculation
        distance = distance_calculator.calculate_distance(mmdbs_image_feature, query_object_feature, distance_function)
        # set distance as value
        mmdbs_image_distance_dictonary['distance'] = distance

        return mmdbs_image_distance_dictonary

    def extract_local_histogram_feature(self, mmdbs_image, seg):

        # extract local histogram feature corresponding to segmentation parameter
        if seg == '2_2':
            mmdbs_image.local_histogram_2_2 = mmdbs_image.extract_histograms(mmdbs_image.image, 2, 2, [8, 2, 4],
                                                                               False)
            return mmdbs_image.local_histogram_2_2['cell_histograms'][0]['values']

        elif seg == '3_3':
            mmdbs_image.local_histogram_3_3 = mmdbs_image.extract_histograms(mmdbs_image.image, 3, 3, [8, 2, 4],
                                                                               False)
            return mmdbs_image.local_histogram_3_3['cell_histograms'][0]['values']

        elif seg == '4_4':
            mmdbs_image.local_histogram_4_4 = mmdbs_image.extract_histograms(mmdbs_image.image, 4, 4, [8, 2, 4],
                                                                               False)
            return mmdbs_image.local_histogram_4_4['cell_histograms'][0]['values']


    def extract_global_histogram_feature(self, mmdbs_image):
        mmdbs_image.global_histogram = mmdbs_image.extract_histograms(mmdbs_image.image, 1, 1, [8, 2, 4], False)
        return mmdbs_image.global_histogram['cell_histograms'][0]['values']


    def extract_global_edge_histogram_feature(self, mmdbs_image):
        mmdbs_image.sobel_edges = mmdbs_image.extract_sobel_edges(mmdbs_image.image)
        min_edge_value = np.min(mmdbs_image.sobel_edges)
        max_edge_value = np.max(mmdbs_image.sobel_edges)
        mmdbs_image.global_edge_histogram = mmdbs_image.extract_histograms_one_channel(mmdbs_image.sobel_edges,
                                                                                         1, 1,
                                                                                         64,
                                                                                         False,
                                                                                         min_edge_value,
                                                                                         max_edge_value)
        return mmdbs_image.global_edge_histogram['cell_histograms'][0]['values']






