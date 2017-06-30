import numpy as np
import load_data
import matplotlib.pyplot as plt

valid_certain = np.load('data/solution_certainties_train_10cat.npy')
pred_results = load_data.load_gz(
    "predictions/final/augmented/valid/try_10cat_wMaxout_next_next_next_next.npy.gz")

valid_max = [max(a) for a in valid_certain]
pred_max = [max(a) for a in pred_results]

print valid_certain.shape
print pred_results.shape

for i, a in enumerate(valid_certain):
    valid_certain[i, np.argmax(a)] = 0.
for i, a in enumerate(pred_results):
    pred_results[i, np.argmax(a)] = 0.


valid_max_2 = [max(a) for a in valid_certain]
pred_max_2 = [max(a) for a in pred_results]


valid_n, valid_bins, valid_patches = plt.hist(
    valid_max, 50, range=(0.2, 1.), normed=True, facecolor='green', alpha=0.75)
pred_n, pred_bins, pred_patches = plt.hist(
    pred_max, 50, range=(0.2, 1.), normed=True, facecolor='red', alpha=0.75)

plt.show()

valid_n, valid_bins, valid_patches = plt.hist(
    valid_max_2, 50, range=(0., 0.6), normed=True, facecolor='green', alpha=0.75)
pred_n, pred_bins, pred_patches = plt.hist(
    pred_max_2, 50, range=(0., 0.6), normed=True, facecolor='red', alpha=0.75)

plt.show()
