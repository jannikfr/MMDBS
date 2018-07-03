import math
from operator import itemgetter
import numpy as np


def calculate_distance(feature1_dic, feature2_dic, distance_function, controller):

    distance_dic = {}

    if distance_function == 'euclidean_distance':
        # Loop over features and sum up the euclidean distances per feature
        for key, value in feature1_dic.items():
            distance_dic[key] = (calculate_euclidean_distance(value, feature2_dic[key]))

    elif distance_function == 'cosine_distance':
        # Loop over features and sum up the Cosine distances per feature
        for key, value in feature1_dic.items():
            distance_dic[key] = calculate_cosine_distance(value, feature2_dic[key])

    elif distance_function == 'quadratic_form_distance':
        # Loop over features and sum up the quadratic form distances per feature
        for key, value in feature1_dic.items():
            # Generate weighting matrix
            # Need to have the same length as the length of the feature vectors
            number_of_attributes = len(get_same_key_set(value, feature2_dic[key])[0])
            weighting_matrix = np.ones((number_of_attributes, number_of_attributes))
            # Calculate distance
            distance_dic[key] = calculate_quadratic_form_distance(value, feature2_dic[key], weighting_matrix)

    elif distance_function == 'weighted_euclidean_distance':
        # Loop over features and sum up weighted euclidean distances per feature
        weighting_matrix = controller.weight_dic
        for key, value in feature1_dic.items():
            weight = weighting_matrix[key]
            distance_dic[key] = calculate_euclidean_distance(value, feature2_dic[key]) * weight
    return distance_dic


def get_same_key_set(a, b):
    """
    Calculate the union set of keys and fill the delta to each dictionary with value 0.
    :param a: A dictionary
    :param b: A dictionary
    :return: The dictionaries a and b enriched with missing keys of the other with value 0.
    """
    # Get the common set of keys
    keys = {**a, **b}.keys()

    # Loop over the set of keys
    for key in keys:
        if key not in a:
            a[key] = 0
        if key not in b:
            b[key] = 0

    return a, b


def calculate_euclidean_distance(feature1, feature2):
    """
    Computes the Euclidean distance between two dictionaries.
    """
    euclidean_distance = 0.0

    feature1, feature2 = get_same_key_set(feature1, feature2)

    # Loop over the set of keys
    for key in feature1.keys():

        feature1_value = feature1[key]
        feature2_value = feature2[key]

        # Calculate the squared difference between the values
        difference = feature1_value - feature2_value
        euclidean_distance = euclidean_distance + difference ** 2

    # Calculate the square root of the sum
    euclidean_distance = math.sqrt(euclidean_distance)
    return euclidean_distance


def calculate_cosine_distance(a, b):
    """
    Computes the Cosine distance between two dictionaries.
    """

    sum_a_times_a = 0.0
    sum_a_times_b = 0.0
    sum_b_times_b = 0.0

    a, b = get_same_key_set(a, b)

    # Loop over the set of keys
    for key in a.keys():
        a_value = a[key]
        b_value = b[key]

        sum_a_times_a = sum_a_times_a + a_value**2
        sum_a_times_b = sum_a_times_b + a_value * b_value
        sum_b_times_b = sum_b_times_b + b_value ** 2

    numerator = sum_a_times_b
    denominator = math.sqrt(sum_a_times_a)*math.sqrt(sum_b_times_b)

    if denominator != 0:
        cosine_distance = 1 - (numerator/denominator)
    else:
        cosine_distance = -1

    # Adjust cosine distance that 1 (best) => 0 and -1 (worst) => 2
    cosine_distance = cosine_distance - 1.0

    return cosine_distance


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
    Computes the Quadratic Form distance between two dictionaries assuming the same key set.
    """

    feature1_dic, feature2_dic = get_same_key_set(feature1_dic, feature2_dic)

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
