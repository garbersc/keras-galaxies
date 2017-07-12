import numpy as np
import os
import csv

with open(TRAIN_LABELS_PATH, 'r') as f:
    reader = csv.reader(f, delimiter=",")
    train_ids = []
    for k, line in enumerate(reader):
        if k == 0: continue # skip header
        train_ids.append(int(line[0]))

isEndClass = np.asarray([0,0,1,
		0,0,
		0,0,
		0,0,
		0,0,0,0,
		0,1,
		0,0,0,
		1,1,1,1,1,1,1,
		0,0,0,
		0,0,0,
		1,1,1,1,1,1])

d = pd.read_csv(TRAIN_LABELS_PATH)
targets = d.as_matrix()[:, 1:].astype('float32')
classes = np.argmax( np.mnultiply(targets,isEndClass)    )



TRAIN_IDS_PATH = "data/train_ids.npy"
# TRAIN_LABELS_PATH = "data/raw/solutions_training.csv"
TRAIN_LABELS_PATH = "data/raw/training_solutions_rev1.csv"

import numpy as np
import os
import csv

with open(TRAIN_LABELS_PATH, 'r') as f:
    reader = csv.reader(f, delimiter=",")
    train_ids = []
    for k, line in enumerate(reader):
        if k == 0: continue # skip header
        train_ids.append(int(line[0]))

train_ids = np.array(train_ids)
print "Saving %s" % TRAIN_IDS_PATH
np.save(TRAIN_IDS_PATH, train_ids)

# TRAIN_LABELS_PATH = "data/raw/solutions_training.csv"
TRAIN_LABELS_PATH = "data/raw/training_solutions_rev1.csv"
TARGET_PATH = "data/solutions_train.npy"

import pandas as pd 
import numpy as np 




d = pd.read_csv(TRAIN_LABELS_PATH)
targets = d.as_matrix()[:, 1:].astype('float32')


print "Saving %s" % TARGET_PATH
np.save(TARGET_PATH, targets)

