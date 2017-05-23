import numpy as np
import matplotlib.pyplot as plt
import json
import matplotlib.lines as mlines

filenames = ['trainingLoss_different_inits_standard_0.txt',
             'trainingLoss_different_inits_standard_standard_1.txt',
             # 'trainingLoss_different_inits.txt',
             'trainingLoss_different_lsuv_0.txt',
             'trainingLoss_different_lsuv_lsuv_1.txt',
             'trainingLoss_different_lsuv_lsuv_preConv_0.txt',
             'trainingLoss_different_lsuv_lsuv_preConv_preConv_1.txt',
             'trainingLoss_different_lsuv_lsuv_preConv_preConv_noNoNorm_0.txt',
             'trainingLoss_different_lsuv_lsuv_preConv_preConv_noNoNorm_noNoNorm_1.txt'
             ]

label_names = ['no Norm 1', 'no Norm 2', 'lsuv 1', 'lsuv 2',
               'pre-training 1 ', 'pre-training 2', 'none 1', 'none 2']

plot_opts = ['r-', 'ro', 'b-', 'bo', 'g-', 'go', 'k-', 'ko']
colors = ['red', 'red', 'blue', 'blue', 'green', 'green', 'black', 'black']

f = [open(i, "r") for i in filenames]
f_lines_ = [f.readlines() for f in f]

dics_ = []

for f_lines in f_lines_:
    print len(f_lines)

    k = 0
    for i in range(len(f_lines)):
        if f_lines[i - k].find("#", 0, 4) >= 0:
            f_lines.remove(f_lines[i - k])
            k += 1

    print 'there are %s non-comment lines in the file' % len(f_lines)

    # clean form non-json
    k = 0
    for i in range(len(f_lines)):
        if f_lines[i - k].find("{", 0, 1) == -1:
            f_lines.remove(f_lines[i - k])
            k += 1

    print 'there are %s json lines in the file' % len(f_lines)

    dics_.append([json.loads(l) for l in f_lines])

trainLoss = []
validLoss = []

for dics in dics_:
    trainLoss.append(dics[-1]["loss"])
    validLoss.append(dics[-2]["loss"])
# validLoss_weighted=data[6]

plots = []
label_h = []

# plt.subplot(211)

for i, loss in enumerate(trainLoss):
    plots.append(plt.plot(xrange(0, len(loss)), loss, plot_opts[i]))
    if i % 2:
        label_h.append(mlines.Line2D(
            [], [], color=colors[i], marker='o',
            markersize=15, linewidth=0, label=label_names[i]))
    else:
        label_h.append(mlines.Line2D(
            [], [], color=colors[i], label=label_names[i]))

plt.legend(handles=label_h,  # bbox_to_anchor=(
           # 0., 1.05),
           loc=1, borderaxespad=0.)

plt.xlabel('mse')
plt.ylabel('epochs')


plt.show()
