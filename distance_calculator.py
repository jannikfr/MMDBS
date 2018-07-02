import math
from operator import itemgetter

import numpy as np



def calculate_distance(feature1_dic, feature2_dic, distance_function, controller):
    distance = 0.0
    if distance_function == 'euclidean_distance':
        for key, value in feature1_dic.items():
            distance = distance + calculate_euclidean_distance(value, feature2_dic[key])
    elif distance_function == 'quadratic_form_distance':
        if len(feature1_dic) > 1:
            weight_matrix = transform_dic_to_matrix_diag(controller.weight_dic)
            distance = calculate_quadratic_form_distance(feature1_dic, feature1_dic, weight_matrix)
        else:
            for key, value in feature1_dic.items():
                distance = distance + calculate_euclidean_distance(value, feature2_dic[key])

    return distance


def calculate_euclidean_distance(feature1, feature2):
    """
    Computes the Euclidean distance between two dictionaries.
    """
    euclidean_distance = 0.0

    # Get the common set of keys
    keys = {**feature1, **feature2}.keys()

    # Loop over the set of keys
    for key in keys:
        feature1_value = 0
        feature2_value = 0
        if key in feature1:
            feature1_value = feature1[key]
        if key in feature2:
            feature2_value = feature2[key]

        # Calculate the squared difference between the values
        difference = feature1_value - feature2_value
        euclidean_distance = euclidean_distance + difference ** 2

    # Calculate the square root of the sum
    euclidean_distance = math.sqrt(euclidean_distance)
    return euclidean_distance


def transform_dic_to_matrix_diag(dic):
    """
    Transforms a dictionary into a matrix with the values on the diagonal, ordered by the key
    :param dic: dictionary of size n
    :return: n times n matrix
    """
    # Transform dic to list of tuples
    tuples = dic.items()
    # Order by former key
    tuples = sorted(tuples, key=itemgetter(0))
    # Remove keys
    values = [i[1] for i in tuples]
    # Create matrix with values in the diagonal
    matrix = np.diag(values)
    return matrix


def transform_dic_to_vector(dic):
    """
     Transforms a dictionary into a vector with the values ordered by the key
     :param dic: dictionary of size n
     :return: vector of size n
     """
    # Transform dic to list of tuples
    tuples = dic.items()
    # Order by former key
    tuples = sorted(tuples, key=itemgetter(0))
    # Remove keys
    values = [i[1] for i in tuples]
    # Create one-dimensional matrix of values
    vector = np.matrix(values)
    return vector


def calculate_quadratic_form_distance(feature1_dic, feature2_dic, weighting_matrix):
    """
    Computes the Euclidean distance between two dictionaries.
    """

    # Transform dictionaries into vectors
    feature1_vec = transform_dic_to_vector(feature1_dic)
    feature2_vec = transform_dic_to_vector(feature2_dic)

    # Substract features
    difference = np.subtract(feature1_vec, feature2_vec)

    # Transpose result
    difference_transposed = np.transpose(difference)

    # Multiply the differences
    sub_multiplied_sub_trans = np.multiply(difference, difference_transposed)

    # Multiply the result with the weighting matrix
    product = np.multiply(sub_multiplied_sub_trans, weighting_matrix)

    # Distance is equal to the square root of the sum
    quadratic_form_distance = np.sqrt(np.sum(product))

    return quadratic_form_distance
