import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


def load_image(infilename):
    img = Image.open(infilename)
    img.load()
    data = np.asarray(img)
    return data


print 'Loading test images'
ti = load_image('test_images/testpic_1.jpg')
print np.shape(ti)
print ti.dtype
ti2 = load_image('test_images/testpic_2.jpg')
print np.shape(ti2)
img = Image.open('test_images/testpic_1.jpg')

canvas, (im1, im2, im3) = plt.subplots(1, 3)
im1.imshow(ti)
im2.imshow(ti2)
im3.imshow(img)
plt.savefig('imageload_test.jpg', dpi=300)
plt.close()

print 'Done!'
