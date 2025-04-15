import ultralytics
from ultralytics import YOLO
import numpy as np
from pathlib import Path
import cv2
from skimage import draw

rr, cc = draw.disk((5, 5), 5)
struct = np.zeros((11,11), dtype=np.uint8)
struct[rr, cc] = 1

path = Path(__file__).parent
model_path = path / "facial_best.pt"

oranges = cv2.imread("oranges.png")

hsv_oranges = cv2.cvtColor(oranges, cv2.COLOR_BGR2HSV)

lower = np.array([10, 240, 210])  # верхняя граница
upper = np.array([15, 255, 255])  # нижняя граница

mask = cv2.inRange(hsv_oranges, lower, upper)  # маска, которая берёт только рыжий цвет
mask = cv2.dilate(mask, struct)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

sorted_contours = sorted(contours, key=cv2.contourArea)
m = cv2.moments(sorted_contours[-1]) # [-1] самый большой контур
# print(m)
# Центроид ищем
cx = int(m["m10"] / m["m00"])
cy = int(m["m01"] / m["m00"])

# print(cx, cy)

bbox = cv2.boundingRect(sorted_contours[-1])

model = YOLO(model_path)

cap = cv2.VideoCapture(0)

while cap.isOpened():

    oranges_copy = oranges.copy()

    ret, frame = cap.read()

    result = model(frame)[0]

    masks = result.masks

    annotated = result.plot()

    if len(masks) == 0:
        continue

    global_mask = masks[0].data.numpy()[0, :, :]

    for mask in masks[1:]:
        global_mask += mask.data.numpy()[0, :, :]

    global_mask = cv2.resize(global_mask, (frame.shape[1],
                                           frame.shape[0])).astype(np.uint8)

    global_mask = cv2.dilate(global_mask, struct)

    gglobal_mask = cv2.bitwise_and(frame, frame, mask=global_mask)

    pos = np.where(global_mask > 0)
    min_y, max_y = int(np.min(pos[0]) * 0.8), int(np.max(pos[0]) * 1.1)
    min_x, max_x = int(np.min(pos[1]) * 0.8), int(np.max(pos[1]) * 1.1)
    global_mask = global_mask[min_y:max_y, min_x:max_x]
    gglobal_mask = gglobal_mask[min_y:max_y, min_x:max_x]

    resized_parts = cv2.resize(gglobal_mask, (bbox[2], bbox[3]), interpolation = cv2.INTER_AREA )
    resized_mask = cv2.resize(global_mask, (bbox[2], bbox[3]), interpolation = cv2.INTER_AREA ) * 255

    x, y, w, h = bbox
    roi = oranges_copy[y:y+h, x:x+w]
    bg = cv2.bitwise_and(roi, roi, mask = cv2.bitwise_not(resized_mask))

    combined_oranges = cv2.add(bg, resized_parts)

    oranges_copy[y: y+h, x: x+w] = combined_oranges

    cv2.imshow("oranges",oranges_copy)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
