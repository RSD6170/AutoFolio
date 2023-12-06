import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scripts.plots import csv_reader


columns = ["instance", "as4mocoRun", "sbsRun", "oracleRun"]
iterations = 6000
cutoff = 3600

df = csv_reader.read_CSV(iterations, columns)

ax = plt.gca()

from cycler import cycler
colormap = plt.get_cmap("gist_ncar")
ax.set_prop_cycle(cycler('color', [colormap(i) for i in np.linspace(0, 0.9, 3)]))

cols = ["as4mocoRun", "sbsRun", "oracleRun"]

min_val = df[cols].min().min()

for col in cols:
    y = df[col].sort_values().cumsum()
    y = np.array(y)
    y[y < min_val] = min_val
    x = [100 * float(i) / len(y) for i in range(len(y))]
    ax.step(y, x, label=col)

ax.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
ax.set_xlabel("Cumulative Runtime [s]")
ax.set_ylabel("% Instances Solved or Timeout")
ax.set_xscale("log")
ax.legend()

plt.show()