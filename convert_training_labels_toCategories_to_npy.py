`# TRAIN_LABELS_PATH = "data/raw/solutions_training.csv"
TRAIN_LABELS_PATH = "data/raw/training_solutions_rev1.csv"
TARGET_PATH = "data/solutions_train_categorised.npy"

import pandas as pd 
import numpy as np 



d = pd.read_csv(TRAIN_LABELS_PATH)
targets = d.as_matrix()[:, 1:].astype('float32')

print targets.shape

questions = [
    targets[:, 0:3], # 1.1 - 1.3,
    targets[:, 3:5], # 2.1 - 2.2
    targets[:, 5:7], # 3.1 - 3.2
    targets[:, 7:9], # 4.1 - 4.2
    targets[:, 9:13], # 5.1 - 5.4
    targets[:, 13:15], # 6.1 - 6.2
    targets[:, 15:18], # 7.1 - 7.3
    targets[:, 18:25], # 8.1 - 8.7
    targets[:, 25:28], # 9.1 - 9.3
    targets[:, 28:31], # 10.1 - 10.3
    targets[:, 31:37], # 11.1 - 11.6
]

questions_cat=[]

for i in xrange(0,len(questions)):
	questions_cat.append( np.asarray([np.zeros( ( len(questions[i][0]) ) , dtype='float32')] * targets.shape[0]) )
	#print questions[i][0]	
	#print questions_cat[i][0]
	#print [0]*len(questions[i][0])
	#print len(questions[i][0])
	#print np.argmax(questions[i][0])
 	#print len(questions_cat[i])
	#print len(questions_cat[i][0])
	for j in xrange(0,targets.shape[0]):
		if questions_cat[0][j][2]: continue #decision tree conditions, applied hard
		if i==1 and not questions_cat[0][j][1]: continue
		if i==2 and not questions_cat[1][j][1]: continue
		if i==3 and not questions_cat[1][j][1]: continue
		if i==4 and not questions_cat[1][j][1]: continue
		if i==6 and not questions_cat[0][j][0]: continue
		if i==9 and not questions_cat[4][j][0]: continue
		if i==10 and not questions_cat[4][j][0]: continue
		#questions_cat[i][j]=np.asarray(questions_cat[i][j],dtype='float32')
		questions_cat[i][j][np.argmax(questions[i][j])]=1.
		#print questions_cat[i][j]
		if np.sum(questions_cat[i][j])!=1: print 'problems with categorisation at %s %s' %(i,j)

targets_cat=np.asarray(questions_cat[0],dtype='float32')
print targets_cat.shape
for i in xrange(1,len(questions_cat)):
	targets_cat=np.hstack((targets_cat,questions_cat[i]))
	print targets_cat.shape


#print targets_cat.shape

print targets_cat

print targets_cat[0]

print targets_cat[5874]


print "Saving %s" % TARGET_PATH
np.save(TARGET_PATH, targets_cat)

