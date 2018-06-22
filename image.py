import numpy
import cv2


class Image:
    def __init__(self):
        self.classification = ""
        self.image = numpy.empty
        self.sobel_edge_detection = numpy.empty
        self.path = ""
        self.global_edge_histogram = {}
        self.local_histogram = {}
        self.global_histogram = {}

    def buildAttributes(self, path, classification):
            self.path = path
            self.classification = classification
            # Convert to HSV color space
            self.image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2HSV)

            return self
