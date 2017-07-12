import numpy as np
import math
import json


def _count_classes(y_valid, cat_option='max'):
    y_valid_ = y_valid
    class_count = {}
    for i in range(len(y_valid_[0])):
        class_count[i] = 0

    if cat_option == 'max':
        for y_i in y_valid_:
            class_count[np.argmax(y_i)] += 1

    return class_count


def _create_class_weight(labels_dict, mu=0.15, savefile='classweights.txt'):
    total = np.sum(labels_dict.values())
    keys = labels_dict.keys()
    class_weight = dict()

    for key in keys:
        score = math.log(mu * total / float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0

    if savefile:
        with open(savefile, 'w') as f:
            json.dump(class_weight, f)

    return class_weight


def create_class_weight(y_valid, cat_option='max', mu=0.15,
                        savefile='classweights.txt'):
    return _create_class_weight(_count_classes(y_valid, cat_option), mu,
                                savefile)
