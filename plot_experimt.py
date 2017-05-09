import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import os


conf = json.load('conf.json')

dim = conf['dim']
gramm = conf['grammar']
name = conf['name']


precisions_files = os.listdir("experiments/precisions_%s_*"%(name))
recalls_files = os.listdir("experiments/precisions_%s_*"%(name))

print(precisions_files)
print(recalls_files)