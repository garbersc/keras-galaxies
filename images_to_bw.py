import skimage
from skimage import io
import os
import glob

IMAGE_PATH = 'data/raw/images_train_rev1/'
IMAGE_BW_PATH = 'data/bw/images_train_rev1/'

if not os.path.isdir('data/bw/'):
    os.mkdir('data/bw/')
if not os.path.isdir(IMAGE_BW_PATH):
    os.mkdir(IMAGE_BW_PATH)

images = glob.glob(
    os.path.join(IMAGE_PATH, "*.jpg"))

for i, img in enumerate(images):
    if not i % 1000:
        print i

    img_ = io.imread(img)
    img_ = skimage.color.rgb2gray(img_)
    img_name = os.path.basename(img)
    io.imsave(IMAGE_BW_PATH + '/' + img_name, img_)


print 'Done!'
