import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.color import rgb2hsv

im = plt.imread("balls_and_rects.png")
im_hsv = rgb2hsv(im)

binary = im.mean(2)
binary[binary > 0] = 1

labeled = label(binary)

regions = regionprops(labeled)


def recognize(region):
    if region.image.mean() == 1.0:
        return "rects"
    else:
        return "balls"


balls_colors = []
rect_colors = []

figures = {
    "balls": 0,
    "rects": 0
}

for region in regions:
    recognized = recognize(region)
    figures[recognized] += 1

    cy, cx = region.centroid
    color = im_hsv[int(cy), int(cx)][0]
    if recognized == 'balls':
        balls_colors.append(round(color, 1))
    else:
        rect_colors.append(round(color, 1))


def colors_counter(figure_colors):
    print("color: amount")
    for fig_color in set(sorted(figure_colors)):
        cnt = 0
        for i in figure_colors:
            if fig_color == i:
                cnt += 1
        print(f"{'%.3f' % fig_color}: {cnt}")
    print('\n')


print(figures)
print("balls colors:")
colors_counter(balls_colors)
print("rect colors:")
colors_counter(rect_colors)
