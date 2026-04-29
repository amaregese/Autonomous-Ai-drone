import cv2

from src.perception.vision_utils.geometry import getCenter, getDelta


def process(output_img):
    gray = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centerpoint = (round(output_img.shape[1] / 2), round(output_img.shape[0] / 2))
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        print(area)
        if area > 100000.0:
            filtered_contours.append(contour)

    count = 0
    target = ((centerpoint[0], centerpoint[1]), 0)
    if len(filtered_contours) > 1:
        for contour in filtered_contours:
            targetCenter = getCenter(contour)
            if count == 0:
                delta = getDelta(targetCenter, centerpoint)
                target = (targetCenter, delta)
            else:
                currentDelta = getDelta(targetCenter, centerpoint)
                if abs(target[1]) > abs(currentDelta):
                    target = (targetCenter, currentDelta)
            count += 1
    elif len(filtered_contours) == 1:
        targetCenter = getCenter(filtered_contours[0])
        delta = getDelta(targetCenter, centerpoint)
        target = (targetCenter, delta)
    else:
        cv2.putText(output_img, "NO TARGET", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)

    cv2.circle(output_img, target[0], 20, (0, 0, 255), thickness=-1, lineType=8, shift=0)
    cv2.putText(output_img, str(target[1]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.line(output_img, centerpoint, target[0], (255, 0, 0), thickness=10, lineType=8, shift=0)
    cv2.circle(output_img, centerpoint, 20, (0, 255, 0), thickness=-1, lineType=8, shift=0)
    cv2.drawContours(output_img, filtered_contours, -1, (255, 255, 255), 3)
    return output_img

