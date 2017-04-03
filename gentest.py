#from custom_for_keras import input_generator
import numpy as np
import load_data
import realtime_augmentation as ra


def input_generator(train_gen):
    for chunk in train_gen:
        if not chunk:
            print 'WARNING: data input generator yielded ' + str(chunk)
            + ', something went wrong'
        chunk_data, chunk_length = chunk
        y_chunk = chunk_data.pop()  # last element is labels.
        xs_chunk = chunk_data

        # need to transpose the chunks to move the 'channels' dimension up
        xs_chunk = [x_chunk.transpose(0, 3, 1, 2) for x_chunk in xs_chunk]

        l0_input_var = xs_chunk[0]
        l0_45_input_var = xs_chunk[1]
        l6_target_var = y_chunk

        yield ([l0_input_var, l0_45_input_var], l6_target_var)


copy_to_ram_beforehand = False

debug = True
predict = False  # not implemented
continueAnalysis = False
saveAtEveryValidation = True

BATCH_SIZE = 1000  # keep in mind

NUM_INPUT_FEATURES = 3

MOMENTUM = 0.9
WEIGHT_DECAY = 0.0
EPOCHS = 150
VALIDATE_EVERY = 20  # 20 # 12 # 6 # 6 # 6 # 5 #
NUM_EPOCHS_NONORM = 0.1
# this should be only a few, just .1 hopefully suffices.

TRAIN_LOSS_SF_PATH = "trainingNmbrs_keras_modular_includeFlip_and_37relu.txt"
# TARGET_PATH = "predictions/final/try_convnet.csv"
WEIGHTS_PATH = "analysis/final/try_convent_keras_modular_includeFlip_and_37relu.h5"

LEARNING_RATE_SCHEDULE = {
    0: 0.4,
    2: 0.1,
    10: 0.05,
    40: 0.01,
    80: 0.005,
    120: 0.0005
    # 500: 0.04,
    # 0: 0.01,
    # 1800: 0.004,
    # 2300: 0.0004,
    # 0: 0.08,
    # 50: 0.04,
    # 2000: 0.008,
    # 3200: 0.0008,
    # 4600: 0.0004,
}
if continueAnalysis:
    LEARNING_RATE_SCHEDULE = {
        0: 0.1,
        20: 0.05,
        40: 0.01,
        80: 0.005
        # 0: 0.0001,
        # 500: 0.002,
        # 800: 0.0004,
        # 3200: 0.0002,
        # 4600: 0.0001,
    }


input_sizes = [(69, 69), (69, 69)]
PART_SIZE = 45

N_INPUT_VARIATION = 2


GEN_BUFFER_SIZE = 2


y_train = np.load("data/solutions_train.npy")
ra.y_train = y_train

# split training data into training + a small validation set
ra.num_train = y_train.shape[0]

# integer division, is defining validation size
ra.num_valid = ra.num_train // 10
ra.num_train -= ra.num_valid


# training num check for EV usage
if ra.num_train != 55420:
    print "num_train = %s not %s" % (ra.num_train, 55420)

ra.y_valid = ra.y_train[ra.num_train:]
ra.y_train = ra.y_train[:ra.num_train]

load_data.num_train = y_train.shape[0]
load_data.train_ids = np.load("data/train_ids.npy")

ra.load_data.num_train = load_data.num_train
ra.load_data.train_ids = load_data.train_ids

ra.valid_ids = load_data.train_ids[ra.num_train:]
ra.train_ids = load_data.train_ids[:ra.num_train]


train_ids = load_data.train_ids
test_ids = load_data.test_ids

num_train = ra.num_train
num_test = len(test_ids)

num_valid = ra.num_valid

y_valid = ra.y_valid
y_train = ra.y_train

valid_ids = ra.valid_ids
train_ids = ra.train_ids

train_indices = np.arange(num_train)
valid_indices = np.arange(num_train, num_train + num_valid)
test_indices = np.arange(num_test)

N_TRAIN = num_train
N_VALID = num_valid


ds_transforms = [
    ra.build_ds_transform(3.0, target_size=input_sizes[0])  # ,
    # ra.build_ds_transform(
    #     3.0, target_size=input_sizes[1])
    # + ra.build_augmentation_transform(rotation=45)
]

num_input_representations = len(ds_transforms)

augmentation_params = {
    'zoom_range': (1.0, 1.),
    'rotation_range': (0, 0),
    'shear_range': (0, 0),
    'translation_range': (-0, 0),
    'do_flip': True,
}

augmented_data_gen = ra.realtime_augmented_data_gen(
    num_chunks=N_TRAIN / BATCH_SIZE * (EPOCHS + 1),
    chunk_size=BATCH_SIZE,
    augmentation_params=augmentation_params,
    ds_transforms=ds_transforms,
    target_sizes=input_sizes)

post_augmented_data_gen = ra.post_augment_brightness_gen(
    augmented_data_gen, std=0.0)

train_gen = load_data.buffered_gen_mp(
    post_augmented_data_gen, buffer_size=GEN_BUFFER_SIZE, sleep_time=2)

input_gen = input_generator(train_gen)


k = 0

print type(augmented_data_gen)

for i in augmented_data_gen:
    if not i:
        print 'i is %s at k=%s' % (i, k)
    elif not k % 100:
        print 'generator running at iteration %s' % k
    k += 1
else:
    print 'iteration stopped after k = %s' % k

# while True
#     if not train_gen.next():
#         print 'generator failed at iteration %s' % k
#     else:
#         if not k % 100:
#             print 'generator running at iteration %s' % k
#     k += 1

    '''
    with input_gen

    with batch size 1000

    generator running at iteration 8000
generator running at iteration 8100
generator running at iteration 8200
generator running at iteration 8300
Traceback (most recent call last):
  File "gentest.py", line 173, in <module>
    if not input_gen.next():
StopIteration

stopped at 8305 iteration

exactly same with train_gen dirctly
exactly same on post_augmented_data_gen
    '''
