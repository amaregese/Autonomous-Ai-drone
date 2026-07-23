import math


def getCenter(contour):
    import cv2

    moments = cv2.moments(contour)
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    return cx, cy


def getDelta(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def get_single_axis_delta(value1, value2):
    return value2 - value1


def point_in_rectangle(point, left, right, top, bottom):
    return left < point[0] < right and top < point[1] < bottom

