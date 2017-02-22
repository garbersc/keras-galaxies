# TRAIN_LABELS_PATH = "data/raw/solutions_training.csv"
TRAIN_LABELS_PATH = "data/raw/training_solutions_rev1.csv"
TARGET_PATH = "data/solutions_train.npy"

import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt
from termcolor import colored
import load_data

output_names=["smooth","featureOrdisk","NoGalaxy","EdgeOnYes","EdgeOnNo","BarYes","BarNo","SpiralYes","SpiralNo","BulgeNo","BulgeJust","BulgeObvious","BulgDominant","OddYes","OddNo","RoundCompletly","RoundBetween","RoundCigar","Ring","Lense","Disturbed","Irregular","Other","Merger","DustLane","BulgeRound","BlulgeBoxy","BulgeNo2","SpiralTight","SpiralMedium","SpiralLoose","Spiral1Arm","Spiral2Arm","Spiral3Arm","Spiral4Arm","SpiralMoreArms","SpiralCantTell"]


#d = pd.read_csv(TRAIN_LABELS_PATH)
#targets = d.as_matrix()[1:, 1:].astype('float32')

targets=load_data.load_gz('predictions/final/augmented/valid/try_convent_continueAt0p02_next.npy.gz')

targets=targets.T

output_corr=np.zeros((37,37))
print targets.shape
for i in xrange(0,37):
    for j in xrange(i,37):
	output_corr[i][j]=np.corrcoef(targets[i],targets[j])[0][1]
	if i!=j and np.abs(output_corr[i][j])>0.3: 
		if np.abs(output_corr[i][j])>0.7: print colored("%s, %s: %s" %(output_names[i],output_names[j],output_corr[i][j]),'green')
		else: print("%s, %s: %s" %(output_names[i],output_names[j],output_corr[i][j]))



plt.imshow(output_corr, interpolation='none')
plt.colorbar()
plt.savefig("targetsCorrelation_valid.jpeg")


#print "Saving %s" % TARGET_PATH
#np.save(TARGET_PATH, targets)

