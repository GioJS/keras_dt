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
# mettere glob.glob!!!
precisions = []
recalls = []
precisions_files = glob.glob("experiments/precisions_%s_*" % (name))
# for file_p in precisions_files:
#     precisions.append(np.loadtxt(file_p))
# plt.plot(np.arange(15), [np.mean(i) for i in precisions])
# plt.show()
recalls_files = glob.glob("experiments/recalls_%s_*" % (name))
print(precisions_files)
print(recalls_files)

for file_p, file_r in zip(precisions_files, recalls_files):
    precisions.append(np.loadtxt(file_p))
    recalls.append(np.loadtxt(file_r))
print(precisions)
print(recalls)

##plots
