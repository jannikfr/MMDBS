import math



def calculate_distance(feature1_dic, feature2_dic, distance_function, controller):
    distance = 0.0
    if distance_function == 'euclidean_distance':
        for key, value in feature1_dic.items():
            distance = distance + calculate_euclidean_distance(value, feature2_dic[key])
    elif distance_function == 'quadratic_form_distance':
        if len(feature1_dic) > 1:
            distance = calculate_quadratic_form_distance(feature1_dic, feature1_dic, controller)
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


def calculate_quadratic_form_distance(feature1_dic, feature2_dic, controller):
    """
    Computes the Euclidean distance between two dictionaries.
    """
    for key, value in feature1_dic.items():
        weight_for_key = get_weight_for_key(key, controller)
        # do crazy shit with value + feature2_dic[key] and weight_for_key



def get_weight_for_key(key, controller):
    if key == 'global_histogram':
        return controller.weight_global_histogram
    if key == 'local_histogram':
        return controller.weight_local_histogram
    if key == 'global_edge_histogram':
        return controller.weight_global_edge_histogram
    if key == 'global_hue_histogram':
        return controller.weight_global_hue_histogram
    if key == 'color_moments':
        return controller.weight_color_moments
    if key == 'central_circle_color_histogram':
        return controller.weight_central_circle_color_histogram
    if key == 'contours':
        return controller.weight_contours