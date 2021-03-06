import cv2
import math
import numpy as np


class MMDBSImage:
    def __init__(self):
        self.classification = ""
        self.path = ""
        self.image = np.empty

        self.local_histogram_2_2 = {}
        self.local_histogram_3_3 = {}
        self.local_histogram_4_4 = {}
        self.global_histogram = {}
        self.sobel_edges = None
        self.global_edge_histogram = {}
        self.global_hue_histogram = {}
        self.color_moments = {}
        self.central_circle_color_histogram = {}
        self.contours = {}

    def set_image(self, path, classification):
        """
        Load image from the given path and sets it with the path and classification as attribute of the object.
        :param path: Path to the image file.
        :param classification: Classification of the image.
        """
        self.path = path
        self.classification = classification

        # Read image from file and convert to HSV color space
        self.image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2HSV)

    @staticmethod
    def extract_histograms(image, h_splits, v_splits, number_of_bins, show_cells):
        """
        Split a given image in equal sized cells corresponding to the given number of vertical and horizontal splits,
        e.g. v_splits = 2 and h_splits=3 results in 6 cells. For each cell a histogram is computed whereby the colors are
        binned corresponding to the parameter number_of_bins.
        :param image: Image whose histograms need to be computed. Assumes normal cv2 color ranges.
        :param h_splits: The number of horizontal splits.
        :param v_splits: The number of vertical splits.
        :param number_of_bins: The number of bins for grouping the subcolor spaced h, s, v
        :param show_cells: If True, the cells are displayed in an external window for 5 seconds each.
        :return: A dictionary containing the h_splits, v_splits, the number_of_splits and the histogram for each cell.
        """

        # Add additional information about histogram computation to the return value
        histogram = {}
        histogram['h_splits'] = h_splits
        histogram['v_splits'] = v_splits
        histogram['number_of_bins (h,s,v)'] = number_of_bins
        histogram['cell_histograms'] = []

        # Split along horizontal axis
        horizontal_split_images = np.array_split(image, h_splits, axis=0)
        for hor_index, horizontal_split_image in enumerate(horizontal_split_images):

            # Loop over split sub images and split each along vertical axis
            horizontal_vertical_split_images = np.array_split(horizontal_split_image, v_splits, axis=1)
            for ver_index, horizontal_vertical_split_image in enumerate(horizontal_vertical_split_images):

                # Display the cells if desired in parameter
                if show_cells:
                    cv2.imshow("Current cell: hor:" + str(hor_index) + "/ver: " + str(ver_index),
                               cv2.cvtColor(horizontal_vertical_split_image, cv2.COLOR_HSV2BGR))
                    cv2.waitKey(5000)

                # Create empty dictonary and add information about histogram of this cell
                cell_histogram = {'horizontal_index': hor_index, 'vertical_index': ver_index, 'values': {}}
                h = 0
                s = 0
                v = 0

                # Loop over each value in pixel and assign bin to the h, s and v values
                for (x, y, z), value in np.ndenumerate(horizontal_vertical_split_image):

                    if z == 0:
                        h = int(value / (180 / number_of_bins[z]))
                    elif z == 1:
                        s = int(value / (256 / number_of_bins[z]))
                    elif z == 2:
                        v = int(value / (256 / number_of_bins[z]))

                        # Build String key in dictionary since needs a String for saving into db
                        key = str(h) + "," + str(s) + "," + str(v)

                        # All values for this pixel are processed
                        # Build key to compute new binned HSV value
                        # Increase the counter for this HSV value if exists
                        if key in cell_histogram['values']:
                            cell_histogram['values'][key] = cell_histogram['values'][key] + 1
                        # HSV value does not exist yet => create it and set counter to 1
                        else:
                            cell_histogram['values'][key] = 1

                        # Reset variables
                        h = 0
                        s = 0
                        v = 0

                # Order dictionary and append it to the overall dictionary
                # cell_histogram = collections.OrderedDict(sorted(cell_histogram.items()))
                histogram['cell_histograms'].append(cell_histogram)

        return histogram

    @staticmethod
    def extract_sobel_edges(image):
        """
        Converts an image to greyscale and detects edges with Sobel filters.
        :param image: Image whose edges needs to be detected.
        :return: np array containing the calculated edges. Array does not have a fixed value range.
        """
        # Convert image to greyscale
        BGR_image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        greyscale_image = cv2.cvtColor(BGR_image, cv2.COLOR_BGR2GRAY)

        # Define the basic kernel
        kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, - 1]], dtype=np.float)

        # Assign kernels for vertical and horizontal edge detection.
        # The kernel for horizontal edge detection is equal to the transposed kernel for vertical edge detection.
        v_kernel = kernel
        h_kernel = np.transpose(kernel)

        # Retrieve the image's dimensions
        N = greyscale_image.shape[0]
        M = greyscale_image.shape[1]

        # Create an empty np array as the output
        output = np.zeros((N, M))

        # Create a additional border of zeros around the greyscale_image
        # Needed for the convolution algorithm
        greyscale_image = np.pad(greyscale_image, pad_width=1, mode='constant', constant_values=0)

        # Loop over rows
        for i in range(1, N - 1):
            # Loop over columns
            for j in range(1, M - 1):
                # Get a 3 to 3 image of the input for the convolution
                sub_input = greyscale_image[(i - 1):(i + 2), (j - 1):(j + 2)]

                # Calcuate the horizontal and the vertical gradient corresponding to the convolution algorithm
                h_gradient = np.sum(np.multiply(sub_input, h_kernel))
                v_gradient = np.sum(np.multiply(sub_input, v_kernel))

                # Compute the direction of the edge
                output[i - 1][j - 1] = math.sqrt(h_gradient * h_gradient + v_gradient * v_gradient)

        return output

    @staticmethod
    def extract_histograms_one_channel(image, h_splits, v_splits, number_of_bins, show_cells, min_value,
                                       max_value):
        """
        Split a given one channel image (e.g. greyscale) in equal sized cells corresponding to the given number of vertical and horizontal splits,
        e.g. v_splits = 2 and h_splits=3 results in 6 cells. For each cell a histogram is computed whereby the colors are
        binned corresponding to the parameter number_of_bins.
        :param image: One channel image whose histograms need to be computed
        :param h_splits: The number of horizontal splits.
        :param v_splits: The number of vertical splits.
        :param number_of_bins: The number of bins for grouping the greyscale range
        :param show_cells: If True, the cells are displayed in an external window for 5 seconds each.
        :param min_value: Smallest value of the color range. Needed for normalization.
        :param max_value: Greatest value of the color range. Needed for normalization.
        :return: A dictionary containing the h_splits, v_splits, the number_of_splits and the histogram for each cell.
        """

        # Add additional information about histogram computation to the return value
        histogram = {}
        histogram['h_splits'] = h_splits
        histogram['v_splits'] = v_splits
        histogram['number_of_bins'] = number_of_bins
        histogram['cell_histograms'] = []

        # Split along horizontal axis
        horizontal_split_images = np.split(image, h_splits, axis=0)
        for hor_index, horizontal_split_image in enumerate(horizontal_split_images):

            # Loop over split sub images and split each along vertical axis
            horizontal_vertical_split_images = np.split(horizontal_split_image, v_splits, axis=1)
            for ver_index, horizontal_vertical_split_image in enumerate(horizontal_vertical_split_images):

                # Display the cells if desired in parameter
                if show_cells:
                    cv2.imshow("Current cell: hor:" + str(hor_index) + "/ver: " + str(ver_index),
                               horizontal_vertical_split_image)
                    cv2.waitKey(5000)

                # Create empty dictionary and add information about histogram of this cell
                cell_histogram = {'horizontal_index': hor_index, 'vertical_index': ver_index, 'values': {}}

                # Loop over each value in pixel and assign bin to the h, s and v values
                for (x, y), value in np.ndenumerate(horizontal_vertical_split_image):

                    if max_value == min_value:
                        normalized_value = 0
                    else:
                        normalized_value = (value - min_value) / (max_value - min_value)
                    if math.isnan(normalized_value):
                        return {"Error_value_position_x": x, "Error_value_position_y": y, "Error_value": value}

                    bin = int(normalized_value * number_of_bins)

                    bin = str(bin)

                    # Increase the counter for this bin if exists
                    if bin in cell_histogram['values']:
                        cell_histogram['values'][bin] = cell_histogram['values'][bin] + 1
                    # HSV value does not exist yet => create it and set counter to 1
                    else:
                        cell_histogram['values'][bin] = 1

                # Order dictionary and append it to the overall dictionary
                # cell_histogram = collections.OrderedDict(sorted(cell_histogram.items()))
                histogram['cell_histograms'].append(cell_histogram)

            return histogram

    @staticmethod
    def extract_color_moments(image):
        """
        Extracts the first three color moments (mean, standard deviation and skewness)
        :param image: Three channel image
        :return: Dictionary containing the three moments for each channel
        """
        color_moments = {}

        for channel_index in range(3):

            channel = image[:, :, channel_index]
            mean = np.mean(channel)
            color_moments["channel_" + str(channel_index) + "_moment_1"] = mean

            sum_moment_2 = 0.0
            sum_moment_3 = 0.0
            for value in np.nditer(channel):
                sum_moment_2 = sum_moment_2 + ((value - mean) ** 2)
                sum_moment_3 = sum_moment_2 + ((value - mean) ** 3)

            color_moments["channel_" + str(channel_index) + "_moment_2"] = (sum_moment_2 / channel.size) ** (
                        1 / float(2))
            color_moments["channel_" + str(channel_index) + "_moment_3"] = (sum_moment_3 / channel.size) ** (
                        1 / float(3))

        return color_moments

    @staticmethod
    def get_central_circle(image):
        """
        Cut out a circle in the center of the image with and radius of 200px of the smallest dimension of the image.
        The outer area is filled white.
        :param image: Original three channel image
        :return: Image with circle and white background.
        """
        # Huge width of the circle's line
        line_width = 1000
        # Calculate the necessary radius of the image
        radius = 100+int(line_width/2)
        # Draw circle
        cv2.circle(image, (int(image.shape[1] / 2), (int(image.shape[0] / 2))), radius, (255, 255, 255), line_width)
        # Convert back to HSV color space
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return image

    @staticmethod
    def extract_contours(image, sobel):
        """
        Extract the 5 biggest contours with their area, arc length and border points.
        :param image: Image object, only needed for displaying purposes.
        :param sobel: Sobel edge data of this image.
        :return: A dictionary containing the 5 biggest contours with their area, arc length and border points.
        """

        contour_data = {}

        # Normalize sobel edges between 0 and 255
        sobel = cv2.normalize(sobel, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F, dst=None)

        # Convert to integers
        sobel = np.uint8(sobel)

        # Display converted sobel image
        # cv2.imwrite("sobel.png", sobel)

        # Apply threshold to image in order to reduce noise
        ret, thresh = cv2.threshold(sobel, 220, 255, cv2.THRESH_TOZERO_INV)

        # Display threshold result
        # cv2.imwrite("thres.png", thresh)

        # Apply canny edge detection. Clears edges.
        canny = cv2.Canny(thresh, 80, 200, 5)

        # Display canny result
        # cv2.imwrite("canny.png", canny)

        # Find contours based on canny edge detection and sobel filters
        _, contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Get 10 biggest contours based on their area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        # Display the 4 biggest contours for demonstration purposes
        # image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        # cv2.drawContours(image, contours, 0, (255, 0, 0), 2)
        # cv2.drawContours(image, contours, 1, (0, 255, 0), 2)
        # cv2.drawContours(image, contours, 2, (0, 0, 255), 2)
        # cv2.drawContours(image, contours, 3, (0, 255, 255), 2)
        # cv2.imwrite("test.png", image)

        # Loop over the selected contours
        for i, contour in enumerate(contours):

            # Calculate area of contour
            area = cv2.contourArea(contour)

            # Calculate the perimeter/arc length of the contour
            # Assuming a closed contour
            perimeter = cv2.arcLength(contour, True)

            # Calculate the number of points, which define this contour
            points_of_contour = len(cv2.approxPolyDP(contour, 0.05 * perimeter, True))

            contour_data[str(i)+'_area'] = area
            contour_data[str(i)+'_perimeter'] = perimeter
            contour_data[str(i)+'_points_of_contour'] = points_of_contour

        return contour_data
