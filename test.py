import math
import cv2
import numpy as np

input_image = cv2.imread("/Users/Jannik/Desktop/MultimediaDatabaseProject/lenna_cropped.jpg")
input_image = cv2.imread("/Users/Jannik/Desktop/MultimediaDatabaseProject/source/chair/image_0001.jpg")


input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, - 1]], dtype=np.float)

v_kernel = kernel
h_kernel = np.transpose(kernel)

N = input_image.shape[0]  # row
M = input_image.shape[1]  # column

output = np.zeros((N, M))

# Surrounds array with 0's on the outside perimeter
input_image = np.pad(input_image, pad_width=1, mode='constant', constant_values=0)

for i in range(1, N - 1):
    for j in range(1, M - 1):
        sub_input = input_image[(i - 1):(i + 2), (j - 1):(j + 2)]

        h_gradient = np.sum(np.multiply(sub_input, h_kernel))
        v_gradient = np.sum(np.multiply(sub_input, v_kernel))

        # Calculate the gradient magnitude
        output[i - 1][j - 1] = math.sqrt(h_gradient * h_gradient + v_gradient * v_gradient)

cv2.imwrite('custom_2d_convolution_gradient.png', output)
