import numpy
import pylab
from PIL import Image
import smplConvLayer as sC
import smplPoolLayer as sP

# open random image of dimensions 639x516
img = Image.open(open('3wolfmoon.jpg'))
# dimensions are (height, width, channel)
img = numpy.asarray(img, dtype='float64') / 256.

# put image in 4D tensor of shape (1, 3, height, width)
img_ = img.transpose(2, 0, 1).reshape(1, 3, 639, 516)
filtered_img = sC.f(img_)
pooled_img = sP.f(filtered_img)
# plot original image and first and second components of output
pylab.subplot(2, 5, 1); pylab.axis('off'); pylab.imshow(img)
#pylab.gray();
# recall that the convOp output (filtered image) is actually a "minibatch",
# of size 1 here, so we take index 0 in the first dimension:
pylab.subplot(2, 5, 2); pylab.axis('off'); pylab.imshow(filtered_img[0, 0, :, :])
pylab.subplot(2, 5, 3); pylab.axis('off'); pylab.imshow(filtered_img[0, 1, :, :])
pylab.subplot(2, 5, 4); pylab.axis('off'); pylab.imshow(filtered_img[0, 2, :, :])
pylab.subplot(2, 5, 5); pylab.axis('off'); pylab.imshow(filtered_img[0, 3, :, :])
pylab.subplot(2, 5, 7); pylab.axis('off'); pylab.imshow(pooled_img[0, 0, :, :])
pylab.subplot(2, 5, 8); pylab.axis('off'); pylab.imshow(pooled_img[0, 1, :, :])
pylab.subplot(2, 5, 9); pylab.axis('off'); pylab.imshow(pooled_img[0, 2, :, :])
pylab.subplot(2, 5, 10); pylab.axis('off'); pylab.imshow(pooled_img[0, 3, :, :])
pylab.show()
