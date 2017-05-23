import numpy as np
import matplotlib.pyplot as plt
import json


# data=np.loadtxt(fname="trainingNmbrs_christmas.txt",delimiter=",")
# data=np.loadtxt(fname="trainingNmbrs_keras_adam_expLR.txt",delimiter=",")
# data=data.transpose()

f = open("trainingNmbrs_geometry_no_maxout_cutOutConv0111.txt", "r")
f_lines = f. readlines()

print len(f_lines)

k = 0
for i in range(len(f_lines)):
    if f_lines[i - k].find("#", 0, 4) >= 0:
        f_lines.remove(f_lines[i - k])
        k += 1

print 'there are %s non-comment lines in the file' % len(f_lines)

# clean form non-json
k = 0
for i in range(len(f_lines)):
    if f_lines[i - k].find("{", 0, 1) == -1:
        f_lines.remove(f_lines[i - k])
        k += 1

print 'there are %s json lines in the file' % len(f_lines)

dics = [json.loads(l) for l in f_lines]

'''
dics=[]
for l in f_lines:
	try:
		dics.append(json.loads(l))
	except ValueError:
		print l
'''

print "found following keys"
for dic in dics:
    print dic.keys()


trainLoss = dics[-1]["loss"]
validLoss = dics[-2]["loss"]
# validLoss_weighted=data[6]

#trainP = plt.plot( rounds, trainLoss, 'ro',label="train")
#validP = plt.plot(rounds, validLoss, 'bo',label="valid")
trainP = plt.plot(xrange(0, len(trainLoss)), trainLoss, 'ro', label="train")
validP = plt.plot(np.array(range(0, len(validLoss))) * len(trainLoss) /
                  len(validLoss), validLoss, 'go', label="valid")
#validP = plt.plot( xrange(0,rounds.shape[0]), validLoss_weighted, 'bo',label="sliced_accuracy")

plt.legend([trainP, validP], ["train", "valid"])

plt.xlabel('Chunks')
plt.ylabel('Mean Square Loss')


plt.show()


# Chunk 480/480
#  load training data onto GPU
#  batch SGD
#  mean training loss (RMSE):            0.098400

# VALIDATING
#  load validation data onto GPU
#  compute losses
#  mean validation loss (RMSE):          0.094632
#  06:53:34 since start (69.95 s)
#  estimated 00:00:00 to go (ETA: Thu Dec 15 01:17:06 2016)
