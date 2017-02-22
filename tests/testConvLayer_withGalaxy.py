import numpy
import pylab
import skimage
from PIL import Image
import smplConvLayer as sC
import smplPoolLayer as sP
import sys
sys.path.append("../kaggle-galaxies/win_sol/kaggle-galaxies/")
from realtime_augmentation import perturb_and_dscrop, build_ds_transform

#def crop(image, xi, yi, xf, yf, *imshowargs, **imshowkwds):
#	#Crop a rectangle; cooridinates 0,0 start at upper left corner
#	img_xf, img_yf = image.shape
#	image = image[yi:yf, xi:xf]
#	return image
#
#def zoom(image, xi, yi, xf, yf, *imshowargs, **imshowkwds):
#	#Plot zoomed-in region of rectangularly cropped image
#	cutimage = crop(image, xi, yi, xf, yf)
#	return imshow(cutimage, *imshowargs, **imshowkwds)

#def post_augment_brightness_gen(data_gen, std=0.5):
#    for target_arrays, chunk_size in data_gen:
#        labels = target_arrays.pop()
#        
#        stds = np.random.randn(chunk_size).astype('float32') * std
#        noise = stds[:, None] * colour_channel_weights[None, :]

#        target_arrays = [np.clip(t + noise[:, None, None, :], 0, 1) for t in target_arrays]
#        target_arrays.append(labels)
#
#        yield target_arrays, chunk_size


img = skimage.io.imread('../kaggle-galaxies/Data/Dev_Img/100134.jpg')
img = img.astype('float32') / 255.0

# open random image of dimensions 639x516
#img = Image.open(open('../kaggle-galaxies/Data/Dev_Img/100134.jpg'))
# dimensions are (height, width, channel)
#img = numpy.asarray(img, dtype='float32') / 256.

# put image in 4D tensor of shape (1, 3, height, width)
img_ = img.transpose(2, 0, 1).reshape(1, 3, 424, 424)
img_cropped=img[108:315,108:315]
#img_cropped = img[108:315][0][108:315][:]
#img_cropped = perturb_and_dscrop(img=img, ds_transforms=build_ds_transform(3.0, target_size=(53, 53)), augmentation_params={
#    'zoom_range': (1.0, 1.1),
#    'rotation_range': (0, 360),
#    'shear_range': (0, 0),
#    'translation_range': (-4, 4),
#}
#, target_sizes=None)
filtered_img = sC.f(img_)
pooled_img = sP.f(filtered_img)
# plot original image and first and second components of output
pylab.subplot(5, 5, 1); pylab.axis('off'); pylab.imshow(img)
#pylab.gray();
# recall that the convOp output (filtered image) is actually a "minibatch",
# of size 1 here, so we take index 0 in the first dimension:
pylab.subplot(5, 5, 2); pylab.axis('off'); pylab.imshow(filtered_img[0, 0, :, :])
pylab.subplot(5, 5, 3); pylab.axis('off'); pylab.imshow(filtered_img[0, 1, :, :])
pylab.subplot(5, 5, 4); pylab.axis('off'); pylab.imshow(filtered_img[0, 2, :, :])
pylab.subplot(5, 5, 5); pylab.axis('off'); pylab.imshow(filtered_img[0, 3, :, :])
pylab.subplot(5, 5, 7); pylab.axis('off'); pylab.imshow(pooled_img[0, 0, :, :])
pylab.subplot(5, 5, 8); pylab.axis('off'); pylab.imshow(pooled_img[0, 1, :, :])
pylab.subplot(5, 5, 9); pylab.axis('off'); pylab.imshow(pooled_img[0, 2, :, :])
pylab.subplot(5, 5, 10); pylab.axis('off'); pylab.imshow(pooled_img[0, 3, :, :])

img_cropped_ = img_cropped.transpose(2, 0, 1).reshape(1, 3, 207, 207)
sP.maxpool_shape = (2, 2)
pooled_crop = sP.f(img_cropped_)

pylab.subplot(5, 5, 11); pylab.axis('off'); pylab.imshow(img_cropped)
pylab.subplot(5, 5, 12); pylab.axis('off'); pylab.imshow(img_cropped_[0,0,:,:])
pylab.subplot(5, 5, 13); pylab.axis('off'); pylab.imshow(img_cropped_[0,1,:,:])
pylab.subplot(5, 5, 14); pylab.axis('off'); pylab.imshow(img_cropped_[0,2,:,:])

pylab.subplot(5, 5, 17); pylab.axis('off'); pylab.imshow(pooled_crop[0, 0, :, :])
pylab.subplot(5, 5, 18); pylab.axis('off'); pylab.imshow(pooled_crop[0, 1, :, :])
pylab.subplot(5, 5, 19); pylab.axis('off'); pylab.imshow(pooled_crop[0, 2, :, :])

img_cNoised=pooled_crop[0, 0, :, :]+skimage.util.random_noise(pooled_crop[0, 0, :, :],var=0.025)
pylab.subplot(5, 5, 20); pylab.axis('off'); pylab.imshow(img_cNoised)
print((img_cNoised).shape)
img_cNoised_= (numpy.asarray([img_cNoised,img_cNoised,img_cNoised])).reshape(1, 3,51,51)
filtered_img_noise=sC.f(img_cNoised_)
lastPool=sP.f(filtered_img_noise)
pylab.subplot(5, 5, 21); pylab.axis('off'); pylab.imshow(lastPool[0, 0, :, :])

img_cNoised_noN= (numpy.asarray([pooled_crop[0, 0, :, :],pooled_crop[0, 0, :, :],pooled_crop[0, 0, :, :]])).reshape(1, 3,51,51)
filtered_img_noisenoN=sC.f(img_cNoised_noN)
lastPoolnoN=sP.f(filtered_img_noisenoN)
pylab.subplot(5, 5, 23); pylab.axis('off'); pylab.imshow(lastPoolnoN[0, 0, :, :])
pylab.show()


def crop(image, xi, yi, xf, yf, *imshowargs, **imshowkwds):
	#Crop a rectangle; cooridinates 0,0 start at upper left corner
	img_xf, img_yf = image.shape
	image = image[yi:yf, xi:xf]
	return image

def zoom(image, xi, yi, xf, yf, *imshowargs, **imshowkwds):
	#Plot zoomed-in region of rectangularly cropped image
	cutimage = crop(image, xi, yi, xf, yf)
	return imshow(cutimage, *imshowargs, **imshowkwds)


