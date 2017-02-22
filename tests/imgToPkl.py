import pickle 
import numpy
import matplotlib.image as mpimg


img=mpimg.imread("image.jpg")
pimg = open("image.pkl","w")
img.resize(1,28*28)
img = img/255.
print(img)
pickle.dump(img,pimg)
pimg.close()
