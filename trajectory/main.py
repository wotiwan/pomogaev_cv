import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
import numpy as np

image = np.load('out/h_0.npy')
labeled = label(image)
regions = regionprops(labeled)

prev_objects = [regions[0], regions[1], regions[2]]

colors = ['red', 'green', 'blue']

for i in range(1, 100):
    im = np.load(f"out/h_{i}.npy")
    labeled = label(im)
    regions = regionprops(labeled)

    cur_objects = [regions[0], regions[1], regions[2]]

    for ind, prev_obj in enumerate(prev_objects):
        distance = 100000
        neighbor = prev_obj
        for cur_obj in cur_objects:
            distance_1 = abs(((cur_obj.centroid[0] - prev_obj.centroid[0]) ** 2
                              + (cur_obj.centroid[1] - prev_obj.centroid[1]) ** 2) ** 0.5)
            if distance_1 < distance:
                distance = distance_1
                neighbor = cur_obj
        plt.plot(neighbor.centroid[1], neighbor.centroid[0], 'o', color=colors[ind], )
        plt.plot((neighbor.centroid[1], prev_obj.centroid[1]),
                 (neighbor.centroid[0], prev_obj.centroid[0]), color=colors[ind])
        prev_objects[ind] = neighbor

plt.show()
