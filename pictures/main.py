import cv2
import numpy as np

my_img_conter = 0


def four_childs(index):
    if hierarchy[0][index][2] != -1:  # У самого большого контура есть child
        index = hierarchy[0][index][2]
        childs_counter = 0
        if hierarchy[0][index][2] != -1:  # у внутренней границы большого контура есть child
            index = hierarchy[0][index][2]
            childs_counter += 1
            while hierarchy[0][index][0] != -1:  # У "окна" есть соседи
                index = hierarchy[0][index][0]
                childs_counter += 1
        return childs_counter
    else:
        return 0


kernel = np.ones((6, 6), np.uint8)

video = cv2.VideoCapture('output.avi')
# cv2.namedWindow("cur_img", cv2.WINDOW_GUI_NORMAL)

ret, frame = video.read()

lower = np.array([0, 0, 60])  # нижняя граница
upper = np.array([50, 50, 150])  # верхняя граница

while frame is not None:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    mask = cv2.dilate(mask, kernel, iterations=5)
    color_found = np.max(mask)

    edges = cv2.Canny(mask, 100, 200)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if color_found > 0 and len(contours) > 0:

        biggest_contour = contours[0]
        for contour in contours:
            if cv2.contourArea(contour) > cv2.contourArea(biggest_contour):
                biggest_contour = contour

        windows_counter = 0
        for ind, contour in enumerate(contours):
            if cv2.contourArea(contour) == cv2.contourArea(biggest_contour):
                windows_counter = four_childs(ind)

        if windows_counter == 4:
            my_img_conter += 1
            # cv2.drawContours(frame, [biggest_contour], -1, (255, 255, 255), 4)
            # cv2.imshow("cur_img", frame)
            # cv2.imshow("cur_img2", edges)
            # print(my_img_conter)

    ret, frame = video.read()

print(my_img_conter)
