import numpy as np
import skimage
import sys
from skimage import io
import warnings
sys.path.append("../kaggle-galaxies/win_sol/kaggle-galaxies/")
#from realtime_augmentation import perturb_and_dscrop, build_ds_transform


inputpaths = [
    'img_10cat/weights_normalized/weight_layer_conv_0_kernel_channel_%i_small.jpg' % i for i in range(3)]
outputpath = 'img_10cat/weights_normalized/weight_layer_conv_0_kernel.jpg'

format = (1, 3)

img = [skimage.io.imread(path_, dtype='float32') for path_ in inputpaths]
if format:
    if len(img) != np.prod(format):
        warnings.warn('format shape does not match image count')
    img = np.array(img)
    img = np.reshape(img, format + img.shape[1:])
    img = np.concatenate(
        img, axis=1)

    img = np.concatenate(
        img, axis=1)
skimage.io.imsave(outputpath, img / 255.)
