import json
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn
import glob

with open('conf.json') as f:
    conf = json.load(f)

dim = conf['dim']
gramm = conf['grammar']
name = conf['name']
med = conf['med'] #per i plot tolgo brutti valori

#mettere glob.glob!!!
precisions_files = glob.glob("experiments/precisions_%s_*"%(name))
recalls_files = glob.glob("experiments/recalls_%s_*"%(name))
print(precisions_files)
print(recalls_files)
##togliere valori med
precisions = []
recalls = []
for file_p, file_r in zip(precisions_files, recalls_files):
    precisions.append(np.loadtxt(file_p))
    recalls.append(np.loadtxt(file_r))
print(precisions)
print(recalls)

##plots


