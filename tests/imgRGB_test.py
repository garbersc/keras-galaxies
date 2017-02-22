#import pickle 
#import numpy
import matplotlib.image as mpimg
#from matplotlib import pyplot as plt
#import pylab

img=mpimg.imread("3wolfmoon.jpg")
print(img)
#plt.imshow(img)
#plt.show()
#pylab.imshow(img)
#pylab.show()
print("-------")
img = img/255.
img.resize(1,516*639,3)
print(img)

