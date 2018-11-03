# Intro OpenCV
import numpy as np
import cv2
import matplotlib.pyplot as plt

def make_coordinatesa(image, line_parameters):
    slope, intercept = line_parameters
    # print(image.shape)
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

def average_slop_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        # print(parameters)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    # print(left_fit)
    # print(right_fit)
    left_fit_average = np.average(left_fit, axis = 0)
    right_fit_average = np.average(right_fit, axis = 0)
    # print(left_fit_average, "left")
    # print(right_fit_average, "right")
    left_line = make_coordinatesa(image, left_fit_average)
    right_line = make_coordinatesa(image, right_fit_average)
    return np.array([left_line, right_line])

def canny(image):
    # Turn the full color to gray
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Smoothening Image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Simple Edge Detection
    canny = cv2.Canny(blur, 50, 150 )
    return canny

def display_lin(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            # print(line)
            # x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


image = cv2.imread("test_image.jpg")
#Turn the image into gray for lane detection.
lane_image = np.copy(image)
canny_image = canny(lane_image)
cropped_image = region_of_interest(canny_image)
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap = 5)

# Optimizing
averaged_lines = average_slop_intercept(lane_image, lines)

# line_image = display_lin(lane_image, lines)
line_image = display_lin(lane_image, averaged_lines)
combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
# print (region_of_interest(canny))

plt.imshow(combo_image)
plt.show()
