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
precisions = {}
recalls = {}
precisions_files = {}
recalls_files = {}
for i in dims:
    precisions_files[i] = glob.glob("experiments/precisions*_%d_m1*" % i)
    recalls_files[i] = glob.glob("experiments/recalls*_%d_m1*" % i)
for i in dims:
    for p, r in zip(precisions_files[i], recalls_files[i]):
        print(p)
        precisions[i] = np.loadtxt(p)
        recalls[i] = np.loadtxt(r)
means_P = []
vars_P = []
means_R = []
vars_R = []
for i in [500]:
    means_P.append(np.mean(precisions[i]))
    vars_P.append(np.var(precisions[i]))
    means_R.append(np.mean(recalls[i]))
    vars_R.append(np.var(recalls[i]))
print(vars_P)

x = np.arange(1)

plt.scatter(x, means_P)
# plt.scatter(x, vars_P)
plt.errorbar(x, means_P, yerr=vars_P)
plt.xticks(x,[500])
plt.ylabel('Mean/Var')
plt.xlabel('Dimension')
plt.show()
# precisions.append(np.mean(precision_f))
# print(precisions)
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
