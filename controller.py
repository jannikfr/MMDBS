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
        mmdbs_image.global_hue_histogram = mmdbs_image.extract_histograms_one_channel(h_image, 1, 1, 64, False, min_h_value, max_h_value)

        return mmdbs_image

    def get_similar_objects(self, query_object, feature, seg):
        """
        Extracts and sets the feature as attributes of the MMDBSImage object.
        :param feature: String identifier of the feature.
        """

        all_mmdbs_images = self.mmdbs_data
        similar_objects = []

        if feature == 'local_histogram':
            if seg == '2_2':
                query_object.local_histogram_2_2 = query_object.extract_histograms(query_object.image, 2, 2, [8, 2, 4],
                                                                                   False)
            elif seg == '3_3':
                query_object.local_histogram_3_3 = query_object.extract_histograms(query_object.image, 3, 3, [8, 2, 4],
                                                                                   False)
            elif seg == '4_4':
                query_object.local_histogram_4_4 = query_object.extract_histograms(query_object.image, 4, 4, [8, 2, 4],
                                                                                   False)

        elif feature == 'global_histogram':

            query_object.global_histogram = query_object.extract_histograms(query_object.image, 1, 1, [8, 2, 4], False)

            for mmdbs_image in all_mmdbs_images:
                temp_mmdbs_image = {}
                temp_mmdbs_image['mmdbs_image'] = mmdbs_image

                mmdbs_image_feature = mmdbs_image.global_histogram['cell_histograms'][0]['values']
                query_object_feature = query_object.global_histogram['cell_histograms'][0]['values']
                distance = distance_calculator.calculate_euclidean_distance(mmdbs_image_feature, query_object_feature)
                temp_mmdbs_image['distance'] = distance

                similar_objects.append(temp_mmdbs_image)

        elif feature == 'global_edge_histogram':
            query_object.sobel_edges = query_object.extract_sobel_edges(query_object.image)
            min_edge_value = np.min(query_object.sobel_edges)
            max_edge_value = np.max(query_object.sobel_edges)
            query_object.global_edge_histogram = query_object.extract_histograms_one_channel(query_object.sobel_edges,
                                                                                             1, 1,
                                                                                             64,
                                                                                             False,
                                                                                             min_edge_value,
                                                                                             max_edge_value)
        similar_objects = sorted(similar_objects, key=itemgetter('distance'))
        return similar_objects

    def get_number_of_mmdbs_images(self):
        return db_connection.get_count_images(self.conn)
