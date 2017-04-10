import pandas as pd 
import numpy as np 

TRAIN_LABELS_PATH = "data/raw/training_solutions_rev1.csv"
TARGET_PATH = "data/solutions_spiralArms_50.npy"
TRAIN_ID_PATH = "data/spiralArms_id50.npy"


d = pd.read_csv(TRAIN_LABELS_PATH)
targets = d.as_matrix()[:,1:].astype('float32')
ids =d.as_matrix()[:,0].astype('float32')

print targets.shape

spiralTargets = np.zeros((targets.shape[0],7),dtype='float32')
spiralIds=np.zeros((ids.shape),dtype='float32')
k=0
for i in xrange(0,targets.shape[0]):
   if targets[i][7]<0.5:
	spiralTargets=np.delete(spiralTargets,(k),axis=0)
	spiralIds=np.delete(spiralIds,(k),axis=0)
   else:
  # spiralTargets[i][0]=targets[i][0]
    	spiralIds[k]=ids[i]
	spiralTargets[k][0]=targets[i][0]+targets[i][2]+targets[i][3]
     	spiralTargets[k][1]=targets[i][31]
     	spiralTargets[k][2]=targets[i][32]
     	spiralTargets[k][3]=targets[i][33]
     	spiralTargets[k][4]=targets[i][34]
     	spiralTargets[k][5]=targets[i][35]
     	spiralTargets[k][6]=targets[i][36]
	k+=1

print spiralTargets.shape

print "Saving %s" % TARGET_PATH
np.save(TARGET_PATH, spiralTargets)
np.save(TRAIN_ID_PATH, spiralIds)
np.savetxt("data/solutions_spiralArms.csv",spiralTargets,delimiter=",")


