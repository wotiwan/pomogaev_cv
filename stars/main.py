import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops

cnt = 0

data = np.load('stars.npy')

labeled = label(data)
regions = regionprops(labeled)

for region in regions:
    if region.image.mean() != 1.0:  # отсеиваем прямоугольники, всё остальное + или x
        cnt += 1

print(cnt)
