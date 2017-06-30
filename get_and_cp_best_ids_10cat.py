import numpy as np
import json
from shutil import copyfile

train_certainty = 'data/solution_certainties_train_10cat.npy'
TRAIN_IDS_PATH = "data/train_ids.npy"

BEST_IDS_PATH = 'best_3_ids_10cat.txt'
IMG_DIR = 'data/raw/images_train_rev1/'
OUT_DIR = 'img_example_10cat/'

output_names = ['round', 'broad_ellipse', 'small_ellipse', 'edge_no_bulge',
                'edge_bulge', 'disc', 'spiral_1_arm', 'spiral_2_arm',
                'spiral_other', 'other']
dtype = []
dtype.append(('img_nr', int))
for q in output_names:
    dtype.append((q, float))

print dtype

train = None


for id, line in zip(np.load(TRAIN_IDS_PATH), np.load(train_certainty)):
    id_line = [id] + list(line)
    if train is None:
        print id_line
        train = np.asarray(tuple(id_line), dtype=dtype)
    else:
        train = np.append(train, np.asarray(tuple(id_line), dtype=dtype))

print np.shape(train)

ten_best_dic = {}
for q in output_names:
    print q
    train_sorted = np.sort(train, order=q)
    ten_best_dic[q] = list(train_sorted['img_nr'][-3:])

with open(BEST_IDS_PATH, 'a') as f:
    json.dump(ten_best_dic, f)
    print '\n'


print 'best ids found, start copying their files...'

dat = ten_best_dic

for name in dat:
    print name
    for i, id in enumerate(dat[name]):
        copyfile(IMG_DIR + str(id) + '.jpg',
                 OUT_DIR + str(name) + '_' + str(i) + '.jpg')

print 'done!'
