import numpy as np
import os
import os.path
import csv
import shutil
import skimage.io
from skimage.transform import downscale_local_mean

TRAIN_IDS_PATH = "data/train_ids.npy"
# TRAIN_LABELS_PATH = "data/raw/solutions_training.csv"
TRAIN_LABELS_PATH = "data/raw/training_solutions_rev1.csv"

dirlabel = 'tf_for_poets/data_cropped'
src_dic = 'data/raw/images_train_rev1/'
n_pix = 69
n_features = 3
cat_10_names = ['round', 'broad_ellipse', 'small_ellipse', 'edge_bulg',
                'edge_no_bulge', 'disc', 'spiral_1_arm', 'spiral_2_arm', 'spiral_other', 'other']

with open(TRAIN_LABELS_PATH, 'r') as f:
    reader = csv.reader(f, delimiter=",")
    train_ids = []
    for k, line in enumerate(reader):
        if k == 0:
            continue  # skip header
        train_ids.append(int(line[0]))

train_ids = np.array(train_ids)

y_train = np.load("data/solutions_train_10cat.npy")

for name in cat_10_names:
    dirname = dirlabel
    dirname = dirname + '/' + name
    if not os.path.isdir(dirname):
        os.mkdir(dirname)

for id, cat in zip(train_ids, y_train):
    n_cat = np.argmax(cat)
    img = np.array(downscale_local_mean(np.array(skimage.io.imread(
        src_dic + str(id) + '.jpg', dtype='float32')), (3, 3, 1)))[
        36:36 + n_pix, 36:36 + n_pix, :n_features]
    skimage.io.imsave(
        dirlabel + '/' + cat_10_names[n_cat] + '/' + str(id) + '.jpg', img / 255.)


for name_ in cat_10_names:
    dirname = dirlabel
    dirname = dirname + '/' + name_ + '/'
    f_c = len(os.listdir(dirname))
    print 'there are in cat ' + name_ + '\t' + str(f_c) + '\tfiles'
