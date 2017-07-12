import numpy as np
import warnings


question_slices = [slice(0, 3), slice(3, 5), slice(5, 7), slice(7, 9),
                   slice(9, 13), slice(13, 15), slice(15, 18), slice(18, 25),
                   slice(25, 28), slice(28, 31), slice(31, 37)]

cat_10_names = ['round', 'broad_ellipse', 'small_ellipse', 'edge_no_bulge',
                'edge_bulge', 'disc', 'spiral_1_arm', 'spiral_2_arm', 'spiral_other', 'other']

cat_10 = [[(0, 0), (6, 0)],
          [(0, 0), (6, 1)],
          [(0, 0), (6, 2)],
          [(0, 1), (1, 0), (8, 2)],
          [(0, 1), (1, 0), (8, 0, 1)],
          [(0, 1), (1, 1), (3, 1)],
          [(0, 1), (1, 1), (3, 0), (10, 0)],
          [(0, 1), (1, 1), (3, 0), (10, 1)],
          [(0, 1), (1, 1), (3, 0), (10, 2, 3, 4, 5)],
          [(0, 2)]
          ]

y_train = np.load("data/solutions_train.npy")

new_sol = []

new_certainty = []
new_certainty_alt = []
new_certainty_alt_2 = []


cert_check = []
not_same_cat = 0
not_same_cat_alt = 0
not_same_cat_alt_2 = 0

wrong_cats = {}
wrong_cats_2 = {}

for i, p in enumerate(y_train):
    p_sol = []
    p_sol += [1 for _ in cat_10]

    cert_alt = []
    cert_alt += [1 for _ in cat_10]

    cert_alt_2 = []
    cert_alt_2 += [1 for _ in cat_10]

    cert = []
    cert += [1 for _ in cat_10]

    for j, cond in enumerate(cat_10):
        for sup_cond in cond:
            combined = False
            cert_add = 0.
            cert_add_alt = 0.
            cert_add_alt_2 = 0.

            p_ = p[question_slices[sup_cond[0]]]
            p_ = [(float(po) / float(np.sum(p_)))
                  if np.sum(p_) else 0. for po in p_]

            for answere in sup_cond[1:]:
                combined = combined or (np.argmax(
                    p[question_slices[sup_cond[0]]]) == answere)
                if sup_cond == cond[-1]:
                    cert_add_alt_2 += p[question_slices[sup_cond[0]]][answere]
                cert_add_alt += p_[answere]
                cert_add += p[question_slices[sup_cond[0]]][answere]
            cert_alt[j] *= cert_add_alt
            p_sol[j] *= combined
            cert[j] *= cert_add
        cert_alt_2[j] = cert_add_alt_2
    if np.sum(p_sol[0:-1]):
        p_sol[-1] = 0
    if np.sum(p_sol) != 1:
        print p_sol
        warnings.warn('categories not exclusive in picture %s' % i)
    new_sol += [p_sol]

    if not np.argmax(p_sol) == np.argmax(cert):
        not_same_cat += 1

    if not np.argmax(p_sol) == np.argmax(cert_alt):
        not_same_cat_alt += 1
        key = '%sto%s' % (np.argmax(p_sol), np.argmax(cert_alt))
        if not key in wrong_cats:
            wrong_cats[key] = 1
        else:
            wrong_cats[key] += 1

    if not np.argmax(p_sol) == np.argmax(cert_alt_2):
        not_same_cat_alt_2 += 1
        key = '%sto%s' % (np.argmax(p_sol), np.argmax(cert_alt_2))
        if not key in wrong_cats_2:
            wrong_cats_2[key] = 1
        else:
            wrong_cats_2[key] += 1

    cert = [c / float(np.sum(cert)) for c in cert]
    cert_alt = [c / float(np.sum(cert_alt)) for c in cert_alt]
    cert_alt_2 = [c / float(np.sum(cert_alt_2)) for c in cert_alt_2]
    new_certainty += [cert]
    new_certainty_alt += [cert_alt]
    new_certainty_alt_2 += [cert_alt_2]

print np.shape(new_sol)
np.save('data/solutions_train_10cat_test.npy', new_sol)
np.save('data/solution_certainties_train_10cat_test.npy', new_certainty)

np.save('data/solution_certainties_train_10cat_alt.npy', new_certainty_alt)
np.save('data/solution_certainties_train_10cat_alt_2.npy', new_certainty_alt_2)

print 'simple maximum and certainty yield not the same category in %s of %s cases' % (not_same_cat, len(y_train))
print 'with norm not the same category in %s of %s cases' % (not_same_cat_alt, len(y_train))
print wrong_cats
print 'with norm not the same category in %s of %s cases' % (not_same_cat_alt_2, len(y_train))
print wrong_cats_2

for k in wrong_cats:
    if not wrong_cats[k] == wrong_cats_2[k]:
        print k
print '\nDone!'
