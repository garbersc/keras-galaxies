import pandas as pd 
import numpy as np 

TRAIN_LABELS_PATH = "data/raw/training_solutions_rev1.csv"
TARGET_PATH = "data/solutions_spiralArms.npy"



d = pd.read_csv(TRAIN_LABELS_PATH)
targets = d.as_matrix()[:,1:].astype('float32')

print targets.shape

spiralTargets = np.zeros((targets.shape[0],7),dtype='float32')
for i in xrange(0,targets.shape[0]):
  # spiralTargets[i][0]=targets[i][0]
   spiralTargets[i][0]=targets[i][0]+targets[i][2]+targets[i][3]
   spiralTargets[i][1]=targets[i][31]
   spiralTargets[i][2]=targets[i][32]
   spiralTargets[i][3]=targets[i][33]
   spiralTargets[i][4]=targets[i][34]
   spiralTargets[i][5]=targets[i][35]
   spiralTargets[i][6]=targets[i][36]

print spiralTargets.shape

print "Saving %s" % TARGET_PATH
np.save(TARGET_PATH, spiralTargets)
np.savetxt("data/solutions_spiralArms.csv",spiralTargets,delimiter=",")


