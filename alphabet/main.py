import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label, regionprops, euler_number
from collections import defaultdict
from pathlib import Path


def recognize(region):
    if region.image.mean() == 1.0:
        return "-"
    else:
        enumber = euler_number(region.image)
        if enumber == -1:  # B or 8
            have_vl = np.sum(np.mean(region.image[:, :region.image.shape[1]//2],
                                     0) == 1) > 3
            if have_vl:
                return "B"
            else:
                return "8"
        elif enumber == 0:  # A or 0
            image = region.image.copy()
            image[-1, :] = 1
            enumber = euler_number(image)
            if enumber == -1:
                return "A"
            else:
                return "0"
        else:  # /, W, X, *, 1
            have_vl = np.sum(np.mean(region.image, 0) == 1) > 3
            if have_vl:
                return "1"
            else:
                if region.eccentricity < 0.4:
                    return "*"
                else:
                    image = region.image.copy()
                    image[0: 1] = 1
                    image[-1, :] = 1
                    image[:, 0] = 1
                    image[:, -1] = 1
                    enumber = euler_number(image)
                    if enumber == -1:
                        return "/"
                    elif enumber == -3:
                        return "X"
                    else:
                        return "W"
    #return "@"


im = plt.imread("alphabet.png")[:, :, :3].mean(2)
im[im > 0] = 1
print(im.shape)
labels = label(im)
print(np.max(labels))
regions = regionprops(labels)

result = defaultdict(lambda: 0)

path = Path("images")

for i, region in enumerate(regions):
    symbol = recognize(region)
    result[symbol] += 1
    # plt.cla()
    # plt.title(f"Symbol - {symbol}")
    # plt.imshow(region.image)
    # plt.savefig(path / f"image_{i:03d}.png")

print(result)
