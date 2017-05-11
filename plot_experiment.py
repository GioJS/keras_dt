import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob





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
        #print(p)
        precisions[i] = np.loadtxt(p)
        recalls[i] = np.loadtxt(r)
means_P = []
vars_P = []
means_R = []
vars_R = []
for i in [500, 1000, 1500]:
    means_P.append(np.mean(precisions[i]))
    vars_P.append(np.std(precisions[i]))
    means_R.append(np.mean(recalls[i]))
    vars_R.append(np.std(recalls[i]))

x = np.arange(3)

plt.scatter(x, means_P)
plt.errorbar(x, means_P, color='#404ee5', ecolor='r', yerr=vars_P, label='Mean Precision')
plt.scatter(x, means_R)
plt.errorbar(x, means_R, color='#49c155', ecolor='r', yerr=vars_R, label='Mean Recall')
plt.legend()

plt.xticks(x, [500, 1000, 1500])
plt.ylabel('Precision/Recall')
plt.xlabel('Dimension')
axes = plt.gca()
axes.set_ylim([0, 1])
plt.title('Grammar M1')
plt.show() #or save
