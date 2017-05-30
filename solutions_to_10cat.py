import numpy as np
import warnings


question_slices = [slice(0, 3), slice(3, 5), slice(5, 7), slice(7, 9),
                   slice(9, 13), slice(13, 15), slice(15, 18), slice(18, 25),
                   slice(25, 28), slice(28, 31), slice(31, 37)]

cat_10_names = ['round', 'broad_ellipse', 'small_ellipse', 'edge_bulg',
                'edge_no_bulge', 'disc', 'spiral_1_arm', 'spiral_2_arm', 'spiral_other', 'other']

cat_10 = [[(0, 0), (6, 0)],
          [(0, 0), (6, 1)],
          [(0, 0), (6, 2)],
          [(0, 1), (1, 0), (8, 2)],
          [(0, 1), (1, 0), (8, 0, 1)],
          [(0, 1), (1, 1), (3, 1)],
          [(0, 1), (1, 1), (3, 0), (10, 0)],
          [(0, 1), (1, 1), (3, 0), (10, 1)],
          [(0, 1), (1, 1), (3, 0), (10, 2, 3, 4, 5)],
          ]

y_train = np.load("data/solutions_train.npy")

new_sol = []

for i, p in enumerate(y_train):
    p_sol = [1]
    p_sol += [1 for _ in cat_10]
    for j, cond in enumerate(cat_10):
        for sup_cond in cond:
            combined = False
            for answere in sup_cond[1:]:
                combined = combined or (np.argmax(
                    p[question_slices[sup_cond[0]]]) == answere)
                if combined:
                    break
            p_sol[j] *= combined
    if np.sum(p_sol[0:-1]):
        p_sol[-1] = 0
    if np.sum(p_sol) != 1:
        print p_sol
        warnings.warn('categories not exclusive in picture %s' % i)
    new_sol += [p_sol]

print np.shape(new_sol)
np.save('data/solutions_train_10cat.npy', new_sol)
print 'Done!'
