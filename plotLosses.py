import numpy as np
import matplotlib.pyplot as plt


#data=np.loadtxt(fname="trainingNmbrs_christmas.txt",delimiter=",")
data=np.loadtxt(fname="trainingNmbrs_keras_maxout_start_w_winsolWeights.txt",delimiter=",")
data=data.transpose()

print("adjusting rounds...")

rounds=data[0]
for i in xrange(1,rounds.shape[0]):
	if rounds[i-1]>rounds[i]:
		rs=rounds.shape[0]
		for j in xrange(i,rs):
			rounds[j]+=rounds[i-1]
		     #	if ( (j<rs-1) and (rounds[j]>rounds[j+1]) ): break
			#rounds[j]+=100

print("done")

trainLoss=data[2]
validLoss=data[3]
validLoss_weighted=data[4]

#trainP = plt.plot( rounds, trainLoss, 'ro',label="train")
#validP = plt.plot(rounds, validLoss, 'bo',label="valid")
trainP = plt.plot(  xrange(0,rounds.shape[0]), trainLoss, 'ro',label="train")
validP = plt.plot( xrange(0,rounds.shape[0]), validLoss, 'go',label="valid")
#validP = plt.plot( xrange(0,rounds.shape[0]), validLoss_weighted, 'bo',label="LL")

plt.legend([trainP, validP],["train","valid"])

plt.xlabel('Chunks')
plt.ylabel('Mean Square Loss')


plt.show()



#Chunk 480/480
#  load training data onto GPU
#  batch SGD
#  mean training loss (RMSE):            0.098400

#VALIDATING
#  load validation data onto GPU
#  compute losses
#  mean validation loss (RMSE):          0.094632
#  06:53:34 since start (69.95 s)
#  estimated 00:00:00 to go (ETA: Thu Dec 15 01:17:06 2016)

