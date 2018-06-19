
working_image = Image()

# working_image.path = "/Users/Jannik/Desktop/MultimediaDatabaseProject/test2.jpg"
#working_image.path = "/Users/Jannik/Desktop/MultimediaDatabaseProject/source/chair/image_0001.jpg"
# working_image.path = "/Users/Jannik/Desktop/MultimediaDatabaseProject/source/gerenuk/image_0031.jpg"
working_image.path = "/Users/Jannik/Desktop/MultimediaDatabaseProject/lenna_cropped.jpg"

working_image.image = cv2.cvtColor(cv2.imread(working_image.path), cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale", working_image.image)
cv2.waitKey(1000)

blur = cv2.GaussianBlur(working_image.image, (15, 15), 0)
cv2.imshow("Blurred", blur)
cv2.waitKey(1000)

bil = cv2.bilateralFilter(working_image.image,15,75,75)
cv2.imshow("Biliteral", bil)
cv2.waitKey(1000)

kernel = numpy.ones((5,5),numpy.uint8)
#working_image.image = cv2.morphologyEx(working_image.image, cv2.MORPH_OPEN, kernel)
eroded = cv2.erode(working_image.image,kernel,iterations = 3)


cv2.imshow("eroded", eroded)
cv2.waitKey(1000)

# kernel = numpy.array([[[1], [2], [1]], [[0], [0], [0]], [[-1], [-2], [- 1]]])
kernel = numpy.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=numpy.float)
kernel2= numpy.array([[4, 2, 0], [0, 0, 0], [-1, -2, - 1]])
print(numpy.multiply(kernel,kernel2))
kernel = numpy.array([[2, 2, 4, 2, 2], [1,1,2,1,1], [0,0,0,0,0], [-1,-1,-2,-1,-1], [-2, -2, -4, -2, -2]])
# yv_kernel = numpy.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, - 1]])
print(kernel)
yv_kernel = numpy.fliplr(kernel)
yh_kernel = numpy.transpose(kernel)

print(yv_kernel)
print(yh_kernel)
# print(yw_kernel)

# h = working_image.image[:,:,2]

input = bil
output = numpy.empty([len(input), len(input[1])], dtype=int)
output = numpy.array(input)
output = numpy.zeros((512, 512))

offset = 2
iterations = 3

input = numpy.pad(input, pad_width=offset, mode='constant', constant_values=0)

for i in range(0, iterations):
    for input_row_index, input_row in enumerate(input[offset:-offset, ]):
        for input_column_index, input_cell in enumerate(input_row[offset:-offset]):
            #subinput = input[input_row_index:input_row_index + 3, input_column_index:input_column_index + 3]
            subinput = input[input_row_index:input_row_index+5, input_column_index:input_column_index+5]
            yv_matrix = (numpy.multiply(subinput, yv_kernel))
            yv = numpy.sum(yv_matrix)

            yv = yv_kernel[0][0] * subinput[0][0] + \
            yv_kernel[0][1] * subinput[0][1] + \
            yv_kernel[0][2] * subinput[0][2] + \
            yv_kernel[1][0] * subinput[1][0] + \
            yv_kernel[1][1] * subinput[1][1] + \
            yv_kernel[1][2] * subinput[1][2] + \
            yv_kernel[2][0] * subinput[2][0] + \
            yv_kernel[2][1] * subinput[2][1] + \
            yv_kernel[2][2] * subinput[2][2]



            yh_matrix = (numpy.multiply(subinput, yh_kernel))
            yh = numpy.sum(yh_matrix)
            result = math.atan2(yv, yh)
            output[input_row_index, input_column_index] = yv


# Detecting edges in the threshold
edge_detected_image = cv2.Canny(input, 75, 200)

# Detecting contours in the edge detecting image
_, contours, hierarchy = cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Setting the closest ball to (-1, -1)
closest_ball = (-1,-1)

# Checking all contours wether they are round and big enough
# to be seen as balls
for contour in contours:

    # Determining the contour approximation
    approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)

    # Calculating the area of the current contour
    area = cv2.contourArea(contour)

    if ((len(approx) > 10) & (100 < area < 3000) ):



        # If the current y-value is higher than the former highest value,
        # the current ball is seen as the current closest ball
        if closest_ball[1] < contour[0][0][1]:
            closest_ball = (contour[0][0][0], contour[0][0][1])
            #cv2.circle(frame,(contour[0][0][0],contour[0][0][1]), 12, (0,0,255), -1)
            cv2.drawContours(input, contour,  -1, (255,0,0), 2)

edged = cv2.Canny(input, 30, 200)

while (1):
    cv2.imshow("Output", output)
    cv2.imwrite('main.png', output)
    cv2.imshow("in", input)
    cv2.imshow("edged", edged)
    k = cv2.waitKey(33)
    if k == 27:  # Esc key to stop
        break

print(len(output))
print(len(output[1]))

# print(j)
# print(working_image.image[i, j, 0])


# convoluted_image = numpy.matmul(h, kernel)

# cv2.imshow("test", convoluted_image)
# cv2.waitKey(10000)

# print(kernel * kernel)
# print(numpy.matmul(kernel, kernel))

values = temp_image.global_histogram['cell_histograms']
print(values[0])
print(sum(values[0].values()))