import numpy
import pylab
import math
import skimage
from PIL import Image
import time
import sys
sys.path.append("../kaggle-galaxies/win_sol/kaggle-galaxies/")
#from realtime_augmentation import perturb_and_dscrop, build_ds_transform

import matplotlib.pyplot as plt

from skimage import data, io, filters, feature, measure, draw
from skimage.morphology import black_tophat, skeletonize, convex_hull_image, watershed
from skimage.transform import downscale_local_mean, hough_ellipse
from skimage.segmentation import random_walker

from scipy import ndimage

from ellipse_fit import get_ellipse_par,points_from_input

#import cv2

img = []

img.append(skimage.io.imread('../kaggle-galaxies/Data/Dev_Img/100008.jpg'))
img.append(skimage.io.imread('../kaggle-galaxies/Data/Dev_Img/100023.jpg'))
img.append(skimage.io.imread('../kaggle-galaxies/Data/Dev_Img/100053.jpg'))
img.append(skimage.io.imread('../kaggle-galaxies/Data/Dev_Img/100078.jpg'))
img.append(skimage.io.imread('../kaggle-galaxies/Data/Dev_Img/100090.jpg'))
img.append(skimage.io.imread('../kaggle-galaxies/Data/Dev_Img/100122.jpg'))
img.append(skimage.io.imread('../kaggle-galaxies/Data/Dev_Img/100123.jpg'))
img.append(skimage.io.imread('../kaggle-galaxies/Data/Dev_Img/100128.jpg'))
img.append(skimage.io.imread('../kaggle-galaxies/Data/Dev_Img/100134.jpg'))
img.append(skimage.io.imread('../kaggle-galaxies/Data/Dev_Img/100143.jpg'))
img.append(skimage.io.imread('../kaggle-galaxies/Data/Dev_Img/100150.jpg'))
img.append(skimage.io.imread('../kaggle-galaxies/Data/Dev_Img/100157.jpg'))
img.append(skimage.io.imread('../kaggle-galaxies/Data/Dev_Img/100187.jpg'))

img.append(skimage.io.imread('../kaggle-galaxies/Data/Dev_Img/698708.jpg'))  
img.append(skimage.io.imread('../kaggle-galaxies/Data/Dev_Img/698744.jpg'))
img.append(skimage.io.imread('../kaggle-galaxies/Data/Dev_Img/698727.jpg')) 
img.append(skimage.io.imread('../kaggle-galaxies/Data/Dev_Img/698761.jpg'))
img.append(skimage.io.imread('../kaggle-galaxies/Data/Dev_Img/698733.jpg'))
img.append(skimage.io.imread('../kaggle-galaxies/Data/Dev_Img/698762.jpg'))


img = numpy.asarray(img,dtype='float32')

img = img / 255.0

n_pix = 69

n_pics = 19

# put image in 4D tensor of shape (1, 3, height, width)
#img_ = img.transpose(0,3, 1, 2).reshape(13, 3, 424, 424)
img_downscaled = downscale_local_mean(img,(1,3,3,1))

img_cropped=img_downscaled[:,36:36+n_pix,36:36+n_pix]


t1=time.time()

img_cropped_unclipped =  img_cropped.transpose(0,3, 1, 2).reshape(n_pics, 3, n_pix, n_pix) 
img_cropped_ = ( numpy.clip( img_cropped.transpose(0,3, 1, 2).reshape(n_pics, 3, n_pix, n_pix) , 0.0, 0.4 ) * 2.5 )

img_cropped_unclipped = numpy.sum(img_cropped_unclipped,1)/3.
img_cropped_ = numpy.sum(img_cropped_,1)/3.


canny=[]
cannys1p5=[]
robert = []
blobs = []
hough = []
ellipse_hand = []
#x_check=[]
#y_check=[]
distances = []
markerses = []
watershed_labels = []
masked_cannies = []

ellipcity_hand = []
ellipcity_hough = []

t_hough =[]
t_candy=[]
t_hand=[]

for i in xrange(0,n_pics):
	#print i
	#blobs.append(feature.blob_doh(img_cropped_[i, 0, :, :], max_sigma=30, threshold=.01))
	img_cropped_unclipped[i,  :, :]/=numpy.amax(img_cropped_unclipped[i])
	'''
	distance = ndimage.distance_transform_edt(img_cropped_unclipped[i,  :, :])
	local_maxi = feature.peak_local_max( distance,min_distance=5, indices=False,  labels=img_cropped_[i,:,:])#footprint=numpy.ones((10, 10)),
	markers = measure.label(local_maxi)
	print markers.reshape(69*69)[numpy.argmax(markers)]
	mask= numpy.asarray(img_cropped_[i,:,:]*5.,dtype='int')
	labels_ws = watershed(-distance, markers, mask= mask)
	distances.append(-distance)	
	markerses.append(markers)
	#markers[~mask] = -1
	#labels_rw = random_walker(img_cropped_[i,:,:], markers)
	watershed_labels.append(labels_ws)
	'''

	tcan = time.time()
	
	if i==0 and False:
	    test0=numpy.zeros(numpy.shape(img_cropped_[0,:,:]))
	    cy, cx = draw.ellipse_perimeter(22, 22, 20, 5, 0.6)
	    test0[cy, cx] = 1.
	    canny.append(test0)
	else:
	    canny.append(feature.canny(img_cropped_[i,  :, :],3))
	t_candy.append(time.time()-tcan)
	'''
	masked_canny = numpy.minimum(canny[i],labels_ws) #numpy.equal(labels_ws,labels_ws[n_pix//2,n_pix//2]))
	masked_cannies.append(masked_canny)
	cannys1p5.append(feature.canny(img_cropped_[i, :, :],1.5))
	'''
	thand =time.time()
	ellipse_hand.append( get_ellipse_par(canny[i] , 1) )
	t_hand.append(time.time()-thand)
	#x_check.append(points_from_input(canny[i],1.,1)[0])
	#y_check.append(points_from_input(canny[i],1.,1)[1])
	#blobs.append(feature.blob_doh(img_cropped_[i, 0, :, :], max_sigma=30, threshold=.01))
	though = time.time()
	ellipses =0 
	ellipses = hough_ellipse(canny[i], accuracy=5, threshold=20,min_size=5, max_size=60)
	if len(ellipses) == 0:  print 'thresh 15';	ellipses = hough_ellipse(canny[i], accuracy=5, threshold=15,min_size=5, max_size=60); 
	if len(ellipses) == 0: print 'thresh 10'; ellipses = hough_ellipse(canny[i], accuracy=5, threshold=10,min_size=5, max_size=60); 
	if len(ellipses) == 0: print 'thresh 5'; ellipses = hough_ellipse(canny[i], accuracy=5, threshold=5,min_size=5, max_size=60); 
	if len(ellipses) == 0: print 'thresh 1'; ellipses = hough_ellipse(canny[i], accuracy=5, threshold=1,min_size=5, max_size=60); 
	if len(ellipses) == 0: print 'min max extremne'; ellipses = hough_ellipse(canny[i], accuracy=5, threshold=1,min_size=1, max_size=70); 
	ellipses.sort(order='accumulator')
	if i == 0: print type(ellipses[-1] )
	hough.append(list(ellipses[-1]))
	
	t_hough.append(time.time()-though)
	

	x,y = points_from_input(canny[i],1.,1)
	s_hand=[]
	s_hough=[]
	s_test=[]
	cosTh=math.cos(-1.*ellipse_hand[i][4])
	sinTh=math.sin(-1.*ellipse_hand[i][4])
	cosTh_ho=math.cos(-1.*hough[i][5])
	sinTh_ho=math.sin(-1.*hough[i][5])
	cosTht=math.cos(-1.*0.6)
	sinTht=math.sin(-1.*0.6)
	for xi,yi in zip(x,y):

		if i == 0: s_test.append( ( (cosTht*(xi-22.) - sinTht*(yi-22.)) / 5. )**2 + ( (sinTht*(xi-22.)+cosTht*(yi-22.)) /20.) **2 )

		#print (((cosTh*(x-ellipse_hand[i][1])-sinTh*(y-ellipse_hand[i][0]))**2)/ellipse_hand[i][3]**2+((sinTh*(x-ellipse_hand[i][0])+cosTh*(y-ellipse_hand[i][1]))**2)/ellipse_hand[i][2]**2)
		s_hand.append((((( ( cosTh*(xi-ellipse_hand[i][1]) - sinTh*(yi-ellipse_hand[i][0]) ) / ellipse_hand[i][3] )**2)+(((sinTh*(xi-ellipse_hand[i][1])+cosTh*(yi-ellipse_hand[i][0]))/ellipse_hand[i][2])**2))-1.)**2)
		s_hough.append(((((cosTh_ho*(xi-hough[i][2])-sinTh_ho*(yi-hough[i][1]))**2)/hough[i][4]**2+((sinTh_ho*(xi-hough[i][2])+cosTh_ho*(yi-hough[i][1]))**2)/hough[i][3]**2)-1.)**2)

	if i == 0: print (numpy.sum(s_test)/len(s_test))**0.5
	if i == 0: print len(x)
	if i == 0: print len(y)
	if i == 0: print cosTh
	if i == 0: print sinTh
	#if i == 0: print ellipse_hand[i]
	ellipcity_hand.append( ( numpy.sum(s_hand)/len(s_hand) )**0.5 )
	ellipcity_hough.append( ( numpy.sum(s_hough)/len(s_hough) )**0.5 ) 
	if i == 0: print len(s_hand)
	if i == 0: print numpy.sum(s_hand)
	#if i == 0: print ellipcity_hand[i]
	

#print len(x_check[0])	

t2=time.time()

print 'needed %4.2fs, that are %3.2fs per image' % (t2-t1 , (t2-t1)/n_pics)
print 'edge finding time per image: %4.3fs' % (numpy.sum(t_candy)/len(t_candy))
print 'quad. eq. fit time per image: %4.3fs' % (numpy.sum(t_hand)/len(t_hand))
print 'hough fit time per image: %4.3fs' % (numpy.sum(t_hough)/len(t_hough))

#print len(ellipses)
#print '\n'
#for ellipse in ellipses: print ellipse[0]
#print '\n'
#print len(hough[0])
#print hough[0]
#print len(hough)


'''
#edges0 = filters.sobel(img_cropped_[0, 0, :, :])
#edges1 = filters.sobel(img_cropped_[0, 1, :, :])
#edges2 = filters.sobel(img_cropped_[0, 2, :, :])

canny0 = feature.canny(img_cropped_[0, 0, :, :])
canny1 = feature.canny(img_cropped_[0, 1, :, :])
canny2 = feature.canny(img_cropped_[0, 2, :, :])

canny0s3 = feature.canny(img_cropped_[0, 0, :, :],1.5)
canny1s3 = feature.canny(img_cropped_[0, 1, :, :],1.5)
canny2s3 = feature.canny(img_cropped_[0, 2, :, :],1.5)


pylab.subplot(5,3, 1); pylab.axis('off'); pylab.imshow(img)


pylab.subplot(5,3, 4); pylab.axis('off'); pylab.imshow(img_cropped_[0, 0, :, :])

pylab.subplot(5,3, 10); pylab.axis('off'); pylab.imshow(canny0)

pylab.subplot(5,3, 13); pylab.axis('off'); pylab.imshow(canny0s3)
'''

pylab.gray()

for i in xrange(0,n_pics):
	pylab.subplot(7,n_pics,i+1); pylab.axis('off'); pylab.imshow(img_cropped_[i,:,:])
	pylab.subplot(7,n_pics,n_pics+i+1); pylab.axis('off'); pylab.imshow(canny[i])
	#pylab.subplot(7,n_pics,2*n_pics+i+1); pylab.axis('off'); pylab.imshow(distances[i])
	#pylab.subplot(7,13,3*13+i+1); pylab.axis('off'); pylab.imshow(markerses[i])
	#pylab.subplot(7,n_pics,4*n_pics+i+1); pylab.axis('off'); pylab.imshow(watershed_labels[i])
	#testimg=numpy.zeros((n_pix,n_pix))
	#xy_check = zip(x_check[i],y_check[i])
	#for x,y in xy_check:
	#	testimg[x,y]=1.
	
	
	#pylab.subplot(7,n_pics,5*n_pics+i+1); pylab.axis('off'); pylab.imshow(masked_cannies[i])

	img2 = img_cropped_unclipped[i,:,:]
	img3 = img_cropped_unclipped[i,:,:]

	pylab.subplot(7,n_pics,3*n_pics+i+1); pylab.axis('off')
	pylab.title('%5.3f' % (s_hand[i] ) ) 
	yc, xc, a, b = [int(round(x)) for x in ellipse_hand[i][0:4]]
	orientation = ellipse_hand[i][4]
	#print yc, xc, a, b, orientation
	cy, cx = draw.ellipse_perimeter(yc, xc, a, b, orientation)
	if a and b:
            for cy1, cx1 in zip(cy,cx):
		if cy1<n_pix and cx1<n_pix:
			img2[cy1,cx1] = 1.
	img2[cy,cx] = 1.
	pylab.imshow(img2)

	pylab.subplot(7,n_pics,6*n_pics+i+1); pylab.axis('off')
	#print hough[i][0]
	pylab.title('%s %5.3f' % (hough[i][0] ,s_hough[i] ) ) 
	yc2, xc2, a2, b2 = [int(round(x)) for x in hough[i][1:5]]
	orientation2 = hough[i][5]
	cy2, cx2 = draw.ellipse_perimeter(yc2, xc2, a2, b2, orientation2)
	#img3 = img_cropped_unclipped[i,0,:,:]
	for cy12, cx12 in zip(cy2,cx2):
		if cy12<n_pix and cx12<n_pix:
			img3[cy12,cx12] = 1.
	pylab.imshow(img3)
		

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


