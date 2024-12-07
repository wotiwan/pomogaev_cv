from skimage.measure import label, regionprops
import cv2

cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cnt_sum = 0

for i in range(1, 13):
    cnt = 0
    img = cv2.imread(f'images/img ({i}).jpg')
    blurred = cv2.GaussianBlur(img, (21, 21), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)

    labeled = label(binary)
    regions = regionprops(labeled)

    for region in regions:
        height, width = region.image.shape[:2]
        if width > height:
            ratio = width / height
        else:
            ratio = height / width
        if 21 > ratio > 12 and region.area > 100000:
            cnt_sum += 1
            cnt += 1
            x1, y1, x2, y2 = region.bbox
            cv2.rectangle(img, (y1, x1), (y2, x2), (255, 0, 50), 5)
        elif region.area > 100000:
            if region.area_bbox / region.area >= 4:
                cnt += 1
                cnt_sum += 1
                x1, y1, x2, y2 = region.bbox
                cv2.rectangle(img, (y1, x1), (y2, x2), (255, 0, 50), 15)

    print(f"image №{i}. pencils found: {cnt}")
    cv2.putText(img, f'pencils count: {cnt}', (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 50), 10)
    cv2.imshow("Frame", img)
    while True:
        key = cv2.waitKey()
        if key == ord('q'):
            break

print(f"Total amount of pencils: {cnt_sum}")
cv2.destroyAllWindows()
