import numpy as np
import os
import csv
import json

TRAIN_LABELS_PATH = "data/raw/training_solutions_rev1.csv"
BEST_IDS_PATH = 'best_9_ids.txt'
output_names = ["smooth", "featureOrdisk", "NoGalaxy", "EdgeOnYes", "EdgeOnNo", "BarYes", "BarNo", "SpiralYes", "SpiralNo", "BulgeNo", "BulgeJust", "BulgeObvious", "BulgDominant", "OddYes", "OddNo", "RoundCompletly", "RoundBetween", "RoundCigar",
                "Ring", "Lense", "Disturbed", "Irregular", "Other", "Merger", "DustLane", "BulgeRound", "BlulgeBoxy", "BulgeNo2", "SpiralTight", "SpiralMedium", "SpiralLoose", "Spiral1Arm", "Spiral2Arm", "Spiral3Arm", "Spiral4Arm", "SpiralMoreArms", "SpiralCantTell"]

dtype = []
dtype.append(('img_nr', int))
for q in output_names:
    dtype.append((q, float))

print dtype

with open(TRAIN_LABELS_PATH, 'r') as f:
    print 'starting csv loop'
    reader = csv.reader(f, delimiter=",")
    train = np.array([], dtype=dtype)
    for k, line in enumerate(reader):
        if not k % 1000:
            print 'line %s' % k
        if k == 0:
            print type(line)
            print np.shape(line)
            continue  # skip header
        elif k == 1:
            train = np.asarray(tuple(line), dtype=dtype)
            continue
        train = np.append(train, np.asarray(tuple(line), dtype=dtype))

print np.shape(train)

ten_best_dic = {}
for q in output_names:
    print q
    train_sorted = np.sort(train, order=q)
    ten_best_dic[q] = list(train_sorted['img_nr'][-9:])

with open(BEST_IDS_PATH, 'a') as f:
    json.dump(ten_best_dic, f)
    print '\n'

print 'done!'
