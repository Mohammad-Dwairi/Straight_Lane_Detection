import cv2 as cv

"""OpenCV is a computer vision and image processing library."""
import numpy as np

"""NumPy is the fundamental package for scientific computing in Python."""


def canny(img):
    """1- Grayscale conversion on the image to reduce color channels and computing time"""
    """2- Smoothing the image by applying GaussianBlur, with 5*5 Kernel (add all the 25 pixels below this kernel, 
          take the average, and replace the central pixel with the new average value. This operation is applied for all 
          the pixels in the image"""
    """3- Canny Edge Detection is a popular edge detection algorithm. Canny(img, low_threshold, high_threshold)
          if the Gradient is larger than the high_threshold then it is accepted as edge pixel.
          if the Gradient is smaller than the low_threshold then it is rejected
          if the Gradient is between the high_threshold and low_threshold then it
          will be accepted only if it is connected to a strong edge"""
    return cv.Canny(cv.GaussianBlur(cv.cvtColor(img, cv.COLOR_RGB2GRAY), (5, 5), 0), 50, 150)


def region_of_interest(img):
    """Used to define the area of interest in the image to search for lines"""

    # returns an array with the same size as the image pixels array with zero values (zero intensity == black image)
    blank_img = np.zeros_like(img)

    height = img.shape[0]
    triangle = np.array([[(200, height), (1100, height), (550, 250)]])
    cv.fillPoly(blank_img, triangle, 255)

    # ANDing the binary value pixels between the canny image and region of interest (produce canny image on the RoI)
    masked_image = cv.bitwise_and(img, blank_img)

    return masked_image


def hough_lines(img):
    """Hough transform algorithm to detect lines in the image by transforming lines as points to the Hough space"""
    return cv.HoughLinesP(img, 2, np.pi / 180, 100, np.array([]), minLineLength=35, maxLineGap=7)


previous_right_avg = []
previous_left_avg = []


def average_lines(h_lines):
    global previous_left_avg, previous_right_avg
    right = []
    left = []
    for line in h_lines:
        x1, y1, x2, y2 = line.reshape(4)
        params = np.polyfit((x1, x2), (y1, y2), 1)
        slope = params[0]
        intercept = params[1]
        if slope < 0:
            left.append((slope, intercept))
        else:
            right.append((slope, intercept))

        if len(left) == 0:
            left_avg = previous_left_avg
        else:
            left_avg = np.average(left, axis=0)
            previous_left_avg = left_avg

        if len(right) == 0:
            right_avg = previous_right_avg
        else:
            right_avg = np.average(right, axis=0)
            previous_right_avg = right_avg

    return left_avg, right_avg


def line_coordinates(img, avg_lines_parameters):
    """finding the line coordinates by it's equation (slope, interception)"""
    print(avg_lines_parameters)
    slope, intercept = avg_lines_parameters
    y1 = img.shape[0]
    y2 = int(y1 * (3 / 5))
    x1 = int((y1 - intercept) / slope)  # depending on the equation y = mx + b
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def display_lines_on_image(img, lanes):
    """Displaying the passed lines on the original picture as the road lanes"""
    blank_img = np.zeros_like(img)
    if lanes is not None:
        for lane in lanes:
            x1, y1, x2, y2 = lane
            cv.line(blank_img, (x1, y1), (x2, y2), (255, 0, 0), 10)

    return cv.addWeighted(img, 0.8, blank_img, 1, 1)


video = cv.VideoCapture("test2.mp4")

while video.isOpened():
    _, frame = video.read()
    canny_image = canny(frame)
    roi = region_of_interest(canny_image)
    lines = hough_lines(roi)
    avg_lines = average_lines(lines)
    left_line_coordinates = line_coordinates(frame, avg_lines[0])
    right_line_coordinates = line_coordinates(frame, avg_lines[1])
    cv.imshow("result", display_lines_on_image(frame, [left_line_coordinates, right_line_coordinates]))
    if cv.waitKey(5) & 0xFF == ord('q'):
        break
video.release()
cv.destroyAllWindows()
