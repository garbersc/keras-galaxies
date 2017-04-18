import json
from shutil import copyfile

BEST_IDS_PATH = 'best_9_ids.txt'
IMG_DIR = 'data/raw/images_train_rev1/'
OUT_DIR = 'img_example/'

print 'reading...'
f = open(BEST_IDS_PATH, "r")
f_lines = f.readlines()

dat = json.loads(f_lines[-1])

for name in dat:
    print name
    for i, id in enumerate(dat[name]):
        copyfile(IMG_DIR + str(id) + '.jpg',
                 OUT_DIR + str(name) + '_' + str(i) + '.jpg')
