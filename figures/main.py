import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops

data = np.load('ps.npy')

labeled = label(data)
obj_count = np.max(labeled)

regions = regionprops(labeled)

cnt_filled = 0
cnt_gaped = 0

for region in regions:
    if region.image.mean() != 1.0:
        cnt_gaped += 1
    else:
        cnt_filled += 1

print(f"""total amount of objects: {obj_count}
rectangles: {cnt_filled}
gaped_rectangles: {cnt_gaped}
""")
