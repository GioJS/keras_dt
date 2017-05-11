import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import glob

with open('conf.json') as f:
    conf = json.load(f)

dim = conf['dim']
gramm = conf['grammar']
name = conf['name']
med = conf['med']  # per i plot tolgo brutti valori
###
# prendere data grammatica
# tutte le precisions e recalls di tutte le dim
# plottare tali dati
###
dims = [500, 1000, 1500, 2000]
precisions = []
recalls = []
precisions_files = {}
for i in dims:
    precisions_files[i] = glob.glob("experiments/precisions*_%d_m1_*" % i)

for i in dims:
    precision_f = []
    for f in precisions_files[i]:
        p = np.loadtxt(f)
        precision_f.append(np.mean(p))
    plt.plot(np.arange(50), precision_f)
    plt.show()
    # precisions.append(np.mean(precision_f))
print(precisions)
# for file_p in precisions_files:
#     precisions.append(np.loadtxt(file_p))
# plt.plot(np.arange(15), [np.mean(i) for i in precisions])
# plt.show()
# recalls_files = glob.glob("experiments/recalls_%s_*" % (name))
# print(precisions_files)
# print(recalls_files)

# for file_p, file_r in zip(precisions_files, recalls_files):
#     precisions.append(np.loadtxt(file_p))
#     recalls.append(np.loadtxt(file_r))
# print(precisions)
# print(recalls)

##plots
