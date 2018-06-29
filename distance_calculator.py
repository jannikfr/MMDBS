import math



def calculate_distance(feature1, feature2, distance_function):
    if distance_function == 'euclidean_distance':
        distance = calculate_euclidean_distance(feature1, feature2)
    elif distance_function == 'quadratic_form_distance':
        distance = calculate_quadratic_form_distance(feature1, feature2)

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


def calculate_quadratic_form_distance(feature1, feature2):
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
