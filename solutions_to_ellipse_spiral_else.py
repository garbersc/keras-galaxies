import numpy as np
import warnings


question_slices = [slice(0, 3), slice(3, 5), slice(5, 7), slice(7, 9),
                   slice(9, 13), slice(13, 15), slice(15, 18), slice(18, 25),
                   slice(25, 28), slice(28, 31), slice(31, 37)]

spiral_or_ellipse_cat = [[(0, 1), (1, 1), (3, 0)], [(0, 1), (1, 0)]]

y_train = np.load("data/solutions_train.npy")

new_sol = []

for i, p in enumerate(y_train):
    p_sol = [1]
    p_sol += [1 for _ in spiral_or_ellipse_cat]
    for j, cond in enumerate(spiral_or_ellipse_cat):
        for sup_cond in cond:
            p_sol[j] *= (np.argmax(
                p[question_slices[sup_cond[0]]]) == sup_cond[1])
    if np.sum(p_sol[0:-1]):
        p_sol[-1] = 0
    if np.sum(p_sol) != 1:
        print p_sol
        warnings.warn('categories not exclusive in picture %s' % i)
    new_sol += [p_sol]

print np.shape(new_sol)
np.save('data/solutions_train_spiral_ellipse_other.npy', new_sol)
print 'Done!'
