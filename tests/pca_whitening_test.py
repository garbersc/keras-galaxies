import numpy as np
import os
import skimage
import pylab
import time
import sys
sys.path.append("../data/raw/images_train_rev1/")
#from realtime_augmentation import perturb_and_dscrop, build_ds_transform

import matplotlib.pyplot as plt

from skimage import data, io, filters, feature, measure, draw
from skimage.transform import downscale_local_mean

#import cv2

#siehe http://cs231n.github.io/neural-networks-2/

#Not used in Convolution Networks

n_pics = 55420

n_pix = 45

n_features=3

img = []

print 'start loading %s pictures' % n_pics


for file_ in os.listdir("../data/raw/images_train_rev1/"):
    if not len(img)%5: 
	print '\r%s / %s' % (len(img),n_pics),
	sys.stdout.flush()
    if len(img)>=n_pics: break
    if file_.endswith(".jpg"):
      img.append(np.array(downscale_local_mean(np.array(skimage.io.imread('../data/raw/images_train_rev1/'+file_),dtype='float32'),(3,3,1)))[36:36+n_pix,36:36+n_pix,:(n_features)].flatten())
print ''
if len(img)<n_pics:
	n_pics=len(img)
	print 'found only %s pictures' % n_pics


'''
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
img.append(list(np.array(skimage.io.imread('/home/garbersc/Downloads/reavers_group.jpg'))[500:500+424,300:300+424]))
'''


def unpickle(file):
     import cPickle
     fo = open(file, 'rb')
     dict = cPickle.load(fo)
     fo.close()
     return dict

def R_G_BtoRGB(x):
	return np.dstack((x[:,:x.shape[1]/3],x[:,x.shape[1]/3:2*x.shape[1]/3],x[:,2*x.shape[1]/3:])).reshape(x.shape)
	

#img = R_G_BtoRGB(unpickle("cifar-10-batches-py/data_batch_1")["data"])

print np.shape(img[0])
print np.shape(img[-1])

img = np.asarray(img,dtype='float32')

print img.shape

img = img / 255.0

n_pics = int(img.shape[0])

n_pix = int(np.sqrt(img.shape[1]/n_features))

print n_pics
print n_pix


# put image in 4D tensor of shape (1, 3, height, width)
#img_ = img.transpose(0,3, 1, 2).reshape(13, 3, 424, 424)

time1=time.time()

#img_downscaled = downscale_local_mean(img,(1,3,3,1))



img_cropped=img#img_downscaled[:,36:36+n_pix,36:36+n_pix,:].reshape(n_pics,n_pix*n_pix*3)

print "cropping took %.2f s" % (time.time()-time1)
time1=time.time()

#zero center
set_mean = np.mean(img_cropped,axis=0)

'''
try:
	np.load('mean_cropped_train_set.npy')
except IOError:
	np.save('mean_cropped_train_set',set_mean)
'''

img_middled = img_cropped #- set_mean

print('--')
print np.max(img_cropped)
print np.min(img_cropped)
print np.mean(img_cropped)
print np.std(img_cropped)
print('--')
print np.max(img_middled)
print np.min(img_middled)
print np.mean(img_middled)
print np.std(img_middled)

print "middeling took %.2f s" % (time.time()-time1)
time1=time.time()


#decorelise via conv matrix
cov = np.dot(img_middled.T,img_middled) / img_middled.shape[0]
print "cov took %.2f s" % (time.time()-time1)
time1=time.time()


try:
	U,S = [np.load('u_formPcaTest_notmiddled_'+str(n_pics)+'_'+str(n_features)+'.npy'),np.load('s_notformPcaTest_notmiddled_'+str(n_pics)+'_'+str(n_features)+'.npy')]
except IOError:
	U,S,V = np.linalg.svd(cov)
	print "svd took %.2f s" % (time.time()-time1)
	print U.shape
	print S.shape
	print V.shape
	np.save('u_formPcaTest_notmiddled_'+str(n_pics)+'_'+str(n_features),np.array(U))
	np.save('s_formPcaTest_notmiddled_'+str(n_pics)+'_'+str(n_features),np.array(S))
'''

imgrot=[]
imgrot_reduced=[]
imgwhite=[]
img_middled=img_middled.reshape(n_pics,n_pix**2,3)
for i in range(n_pics):
	if not i%5: 
		print '\r%s / %s' % (i,n_pics),
		sys.stdout.flush()
	cov=np.dot(img_middled[i].T,img_middled[i]) / img_middled[i].shape[0]
	U,S,V = np.linalg.svd(cov)
	
	imgrot.append(np.dot(img_middled[i], U))

	imgrot_reduced.append(np.dot(img_middled[i], U[:,:1]))

	#whiten
	imgwhite.append(imgrot[i] / np.sqrt(S + 1e-5))

print "svd took %.2f s" % (time.time()-time1)
print U.shape
print imgrot_reduced[0].shape

imgrot=np.array(imgrot)
imgrot_reduced=np.array(imgrot_reduced)
imgwhite=np.array(imgwhite)
'''

print U.shape

time1=time.time()
imgrot = np.array([np.dot( p , U) for p in img_middled ])
print 'rotated'

imgrot_reduced = np.array([np.dot( p , U[:,:150]) for p in img_middled ])
print 'reduced'

imgrot_reduced = np.array([np.dot( p , U[:,:150].T) for p in imgrot_reduced ])
print 'reduced_rb'

imgrb = np.array([np.dot( p , U.T) for p in imgrot ])
print 'rb'

#whiten
imgwhite = imgrot / np.sqrt(S + 1e-5)
print 'whitened'
#rotate back in img space
imgwhite = np.array([np.dot( p , U.T) for p in imgwhite ])
print 'whitened_rb'
#imgwhite = np.dot(imgwhite, U.T)

print "u and s ops took %.2f s" % (time.time()-time1)

print('--')
print np.max(imgrot)
print np.min(imgrot)
print np.mean(imgrot)
print np.std(imgrot)
print('--')
print np.max(imgrot_reduced)
print np.min(imgrot_reduced)
print np.mean(imgrot_reduced)
print np.std(imgrot_reduced)
print('--')
print np.max(imgwhite)
print np.min(imgwhite)
print np.mean(imgwhite)
print np.std(imgwhite)

#pylab.subplot(5,3, 1); pylab.axis('off'); pylab.imshow(img)
#pylab.subplot(5,3, 4); pylab.axis('off'); pylab.imshow(img_cropped_[0, 0, :, :])

def renorm(y):
	x=np.array(y)
	for i in range(len(x)):
		x[i]=x[i]-np.min(x[i]) 
		x[i]=x[i]/(np.max(x[i]))
		#x[i].reshape(np.shape(y[i]))
	return x

if n_features==3:
	imgrb_ =  imgrb.reshape(n_pics,n_pix,n_pix,n_features)
	img_cropped_ = img_cropped.reshape(n_pics,n_pix,n_pix,n_features)
	img_middled_= renorm(img_middled.reshape(n_pics,n_pix,n_pix,3))
	imgrot_= renorm(imgrot.reshape(n_pics,n_pix,n_pix,n_features))
	imgrot_reduced_ = renorm(imgrot_reduced.reshape(n_pics,int(np.sqrt(imgrot_reduced.size/n_pics/n_features)),int(np.sqrt(imgrot_reduced.size/n_pics/n_features)),n_features))
	imgwhite_= renorm(imgwhite.reshape(n_pics,n_pix,n_pix,n_features))
elif n_features==1:
	img_cropped_ = img_cropped.reshape(n_pics,n_pix,n_pix)
	#img_middled_= renorm(img_middled.reshape(n_pics,n_pix,n_pix))
	imgrot_= renorm(imgrot.reshape(n_pics,n_pix,n_pix))
	imgrot_reduced_ = renorm(imgrot_reduced.reshape(n_pics,int(np.sqrt(imgrot_reduced.size/n_pics/n_features)),int(np.sqrt(imgrot_reduced.size/n_pics/n_features))))
	imgwhite_= renorm(imgwhite.reshape(n_pics,n_pix,n_pix))
else: print '%s fearures not plotable' % n_features

Urarr =renorm( U.T.reshape(U.shape[0],n_pix,n_pix,n_features))




print imgrot_reduced_.shape

ev95 = []
for i in range(n_pics):
	if not n_pics%5: 
		print '\r%s / %s' % (i,n_pics),
		sys.stdout.flush()
	imgrotnorm=np.fabs(imgrot[i])/np.sum(np.fabs(imgrot[i]))
	imgrotnormgen = (j for j in imgrotnorm)
	c=0
	frakt=0
	while frakt<=0.95:
		c+=1
		frakt+=imgrotnormgen.next()
	ev95.append(c)
np.save('ev95',ev95)

print ev95
print np.min(ev95)
print np.max(ev95)
print np.std(ev95)

print float(np.min(ev95))/n_pix/n_pix/n_features
print float(np.max(ev95))/n_pix/n_pix/n_features
print float(np.std(ev95))/n_pix/n_pix/n_features



ev99 = []
for i in range(n_pics):
	if not n_pics%5: 
		print '\r%s / %s' % (i,n_pics),
		sys.stdout.flush()
	imgrotnorm=np.fabs(imgrot[i])/np.sum(np.fabs(imgrot[i]))
	imgrotnormgen = (j for j in imgrotnorm)
	c=0
	frakt=0
	while frakt<=0.99:
		c+=1
		frakt+=imgrotnormgen.next()
	ev99.append(c)
np.save('ev99',ev99)

print ev99
print np.min(ev99)
print np.max(ev99)
print np.std(ev99)

print float(np.min(ev99))/n_pix/n_pix/n_features
print float(np.max(ev99))/n_pix/n_pix/n_features
print float(np.std(ev99))/n_pix/n_pix/n_features



n_pics=np.min([20,n_pics])
for i in xrange(0,n_pics):
	pylab.subplot(12,n_pics,i+1); pylab.axis('off'); pylab.imshow(img_cropped_[i], interpolation='None')
	pylab.subplot(12,n_pics,n_pics+i+1); pylab.axis('off'); pylab.imshow(img_middled_[i], interpolation='None')
	pylab.subplot(12,n_pics,2*n_pics+i+1); pylab.axis('off'); pylab.imshow(imgrot_[i], interpolation='None')
	pylab.subplot(12,n_pics,3*n_pics+i+1); pylab.axis('off'); pylab.imshow(imgrot_reduced_[i], interpolation='None')
	pylab.subplot(12,n_pics,4*n_pics+i+1); pylab.axis('off'); pylab.imshow(imgwhite_[i], interpolation='None')
	try: pylab.subplot(12,n_pics,6*n_pics+i+1); pylab.axis('off'); pylab.imshow(imgrot_[i,:,:,0], interpolation='None') 
	except IndexError: pass
	try: pylab.subplot(12,n_pics,7*n_pics+i+1); pylab.axis('off'); pylab.imshow(imgrot_reduced_[i,:,:,0], interpolation='None')
	except IndexError: pass
	try: pylab.subplot(12,n_pics,8*n_pics+i+1); pylab.axis('off'); pylab.imshow(imgwhite_[i,:,:,0], interpolation='None')
	except IndexError: pass
	pylab.subplot(12,n_pics,9*n_pics+i+1); pylab.axis('off'); pylab.imshow(Urarr[i], interpolation='None')
	pylab.subplot(12,n_pics,10*n_pics+i+1); pylab.axis('off'); pylab.imshow(imgrb_[i], interpolation='None')
	#pylab.subplot(12,n_pics,11*n_pics+i+1); pylab.axis('off'); pylab.imshow(imgrb_[i]+set_mean.reshape(n_pix,n_pix,n_features), interpolation='None')

if n_features==3: pylab.subplot(12,n_pics,5*n_pics+1); pylab.axis('off'); pylab.imshow(set_mean.reshape(n_pix,n_pix,n_features), interpolation='None')
else: pylab.subplot(12,n_pics,5*n_pics+1); pylab.axis('off'); pylab.imshow(set_mean.reshape(n_pix,n_pix), interpolation='None')		

pylab.savefig('psa_whitened_galaxies_wBackRot_notmiddeled_best24eigenvectors_muchinput.pdf')

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



