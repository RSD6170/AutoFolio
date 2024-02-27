import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scripts.plots import csv_reader

# from https://github.com/mlindauer/asapy/blob/master/asapy/utils/util_funcs.py
def get_cdf_x_y(data, cutoff):
    b_x, b_y, i_s = [], [], 0
    for i, x in enumerate(np.sort(data)):
        b_x.append(x)
        if x < cutoff:
            b_y.append(100 * float(i) /len(data))
            i_s = i
        else:
            b_y.append(100 * float(i_s) /len(data))
    return b_x, b_y


iterations = 8000
cutoff = 3600

df = csv_reader.read_CSV(iterations)

ax = plt.gca()

from cycler import cycler
colormap = plt.get_cmap("gist_ncar")
ax.set_prop_cycle(cycler('color', [colormap(i) for i in np.linspace(0, 0.9, 3)]))

cols = [ ("as4mocoRun", "as4moco"), ("sbsRun", "SBS"), ("oracleRun", "Oracle")]
col2 = [ "as4mocoRun","sbsRun","oracleRun"]

min_val = df[col2].min().min()

for col, lab in cols:
    x, y = get_cdf_x_y(df[col], 3600)
    x = np.array(x)
    x[x < min_val] = min_val
    y = np.array(y)
    ax.step(x, y, label=lab)

ax.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
ax.set_xlabel("Runtime [s] (logarithmic)")
ax.set_ylabel("% Instances Solved")
ax.set_xscale("log")
ax.set_xlim([min_val, cutoff])
ax.set_ylim([0, 100])
ax.legend()

plt.show()
