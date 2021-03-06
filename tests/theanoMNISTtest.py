import cPickle, gzip, numpy

# Load the dataset
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

print(train_set[0])
print "-----"
print(train_set[0][0])
print "-----"
print(train_set[0][1])
print "-----"
print(train_set[1])
print "-----"
print(train_set[1][0])
#import matplotlib.pyplot as plt       # think this is necessary...haven't tested
#plt.imshow(train_set[])
#train_set[1].show() # show the figure, edit it etc.!

from matplotlib import pyplot as plt

dataset=open("image.pkl","r")
datasets = cPickle.load(dataset)
imgpkl = datasets
imgpkl.resize(28,28)
plt.imshow(imgpkl)
plt.show()

#plt.imshow(train_set[0], interpolation='nearest')
a=test_set[0][0]
a.resize(28,28)
plt.imshow(a)
plt.show()

b=test_set[0][1]
b.resize(28,28)
plt.imshow(b)
plt.show()
