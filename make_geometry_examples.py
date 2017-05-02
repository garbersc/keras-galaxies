import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.draw import ellipse, polygon, circle

PATH = "geometry_examples"
if not os.path.isdir(PATH):
    os.mkdir(PATH)

nr = range(1000)
solutions = []

for i in nr:
    sol = [1, 0, 0]
    np.random.shuffle(sol)
    solutions += [sol]

    if not i % 100:
        print i
        print solutions[i]

    img = np.zeros((424, 424, 3), dtype=np.double)
    if np.argmax(sol) == 0:
        rr, cc = ellipse(np.random.randint(200, 224), np.random.randint(
            200, 224), np.random.randint(10, 106), np.random.randint(106, 200),
            img.shape)
    elif np.argmax(sol) == 1:
        rr, cc = circle(np.random.randint(200, 224), np.random.randint(
            200, 224), np.random.randint(10, 150), img.shape)
    else:
        a = np.random.randint(106, 212)
        b = np.random.randint(106, 212)
        c = np.random.randint(213, 321)
        d = np.random.randint(213, 321)
        poly = np.array((
            (a, b),
            (a, c),
            (d, c),
            (d, b),
        ))
        rr, cc = polygon(poly[:, 0], poly[:, 1], img.shape)

    img[rr, cc, :] = 1

    # plt.figure(figsize=(5, 5))
    # plt.axis('off')
    # plt.imshow(img)
    plt.imsave(PATH + '/' + str(i) + '.jpg', img)
    # bbox_inches='tight', pad_inches=0.,)
    # facecolor='b', edgecolor='b')
    # plt.close()

np.save(PATH + '/ids.npy', nr)
np.save(PATH + '/solutions.npy', solutions)
