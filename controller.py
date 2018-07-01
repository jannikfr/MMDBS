from operator import itemgetter
import numpy as np
import db_connection
import distance_calculator
import matplotlib.pyplot as plt


class Controller(object):

    def __init__(self):
        self.conn = db_connection.connect()
        self.mmdbs_data = db_connection.get_all_images(self.conn)

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

        elif feature == 'global_hue_histogram':
            query_object_feature = self.extract_global_hue_histogram_feature(query_object)

        elif feature == 'color_moments':
            query_object_feature = query_object.extract_color_moments(query_object)

        elif feature == 'central_circle_color_histogram':
            query_object_feature = self.extract_central_circle_color_histogram_feature(query_object)

        elif feature == 'contours':
            query_object_feature = self.extract_contours_feature(query_object)

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
            similar_objects.append(
                self.get_distance(query_object_feature, feature, seg, distance_function, mmdbs_image))

        return similar_objects

    def get_number_of_mmdbs_images(self):
        return db_connection.get_count_images(self.conn)

    def get_distance(self, query_object_feature, feature, seg, distance_function, mmdbs_image):
        # read the selected feature from the mmdbs_image (database object)
        mmdbs_image_feature = self.get_mmdbs_image_feature(feature, seg, mmdbs_image)
        # build the mmdbs_image_distance dic with mmdbs_image:distance
        return self.get_mmdbs_image_distance_dictionary(mmdbs_image_feature, query_object_feature, distance_function,
                                                        mmdbs_image, feature)

    def get_mmdbs_image_feature(self, feature, seg, mmdbs_image):
        # read the selected feature from the mmdbs_image
        if feature == 'global_histogram':
            return mmdbs_image.global_histogram

        elif feature == 'global_edge_histogram':
            return mmdbs_image.global_edge_histogram

        elif feature == 'local_histogram':
            if seg == '2_2':
                return mmdbs_image.local_histogram_2_2

            elif seg == '3_3':
                return mmdbs_image.local_histogram_3_3

            elif seg == '4_4':
                return mmdbs_image.local_histogram_4_4

        elif feature == 'global_hue_histogram':
            return mmdbs_image.global_hue_histogram

        elif feature == 'color_moments':
            return mmdbs_image.color_moments

        elif feature == 'central_circle_color_histogram':
            return mmdbs_image.central_circle_color_histogram

        elif feature == 'contours':
            return mmdbs_image.contours

    def get_mmdbs_image_distance_dictionary(self, mmdbs_image_feature, query_object_feature, distance_function,
                                            mmdbs_image, feature):
        # initialize dic
        mmdbs_image_distance_dictonary = {}
        # set image as key
        mmdbs_image_distance_dictonary['mmdbs_image'] = mmdbs_image
        # get feature value for distance calculation
        mmdbs_image_feature_value = self.get_value_for_distance_calculation(mmdbs_image_feature, feature)
        query_object_feature_value = self.get_value_for_distance_calculation(query_object_feature, feature)

        # call distance function for calculation
        distance = distance_calculator.calculate_distance(mmdbs_image_feature_value, query_object_feature_value,
                                                          distance_function)
        # set distance as value
        mmdbs_image_distance_dictonary['distance'] = distance

        return mmdbs_image_distance_dictonary

    def get_value_for_distance_calculation(self, mmdbs_image_feature, feature):
        # choose correct value for calculation dependent on feature...temporary all do the same
        mmdbs_image_feature = mmdbs_image_feature['cell_histograms'][0]['values']
        return mmdbs_image_feature

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

        relevant = [0]
        precision = []
        recall = []

        correct_objects = 0

        for result_object in similar_objects[:number_of_results]:
            result_object_classification = result_object['mmdbs_image'].classification

            if result_object_classification == correct_classification:
                correct_objects = correct_objects + 1
                relevant.append(relevant[-1]+1)
            else:
                relevant.append(relevant[-1])

        relevant = relevant[1:number_of_results]
        for i, value in enumerate(relevant):
            k = i + 1
            precision.append(value/k)
            recall.append(value/correct_objects)

        fig, ax = plt.subplots()
        ax.set_title(r'Precision-Recall-Curve of the '+str(number_of_results)+' best results.')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')

        fig.tight_layout()
        plt.plot(recall, precision, marker='o', markersize=5)
        plt.axis([0, 1.1, 0, 1.1])
        plt.savefig('precision_recall.png')


