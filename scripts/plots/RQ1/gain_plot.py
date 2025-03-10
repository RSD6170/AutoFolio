import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scripts.plots import csv_reader
import matplotlib.patches as mpatches

columns = ["instance", "as4mocoRun", "sbsRun", "oracleRun"]
iterations = 4000

df = csv_reader.read_CSV(iterations, columns)
df = df.assign(instance = lambda x: x['instance'].str.extract('mc2022_track1_(\d+).dimacs'))
df["instance"] = df["instance"].astype(int)
#df.set_index("instance", inplace=True)

df["gain"] = df.apply(lambda row: ((row.as4mocoRun / row.sbsRun)), axis=1)
mean_log = df["gain"].apply(lambda row: math.log10(row)).mean()
mean = 10 ** mean_log
median = df["gain"].median()
df["color"] = df.apply(lambda row: "r" if row.gain >= 1 else "g", axis=1)
df["plot"] = df.apply(lambda row: row.gain - 1, axis=1)

print(mean)
print(median)

ax = plt.gca()
plt.bar(df["instance"], df["plot"], color=df["color"], bottom=1, log=True)
ax.axhline(1, color='black')
ax.axhline(mean, color='b', ls='--', label="Mean")
ax.axhline(median, color='b', label="Median")


plt.ylabel("Score -- as4moco / SBS")
plt.xlabel("Identifier of input instance")

# https://stackoverflow.com/a/56551701
handles, labels = ax.get_legend_handles_labels()
better = mpatches.Patch(color='g', label='Better')
worse = mpatches.Patch(color='r', label='Worse')

handles.append(better)
handles.append(worse)

plt.legend(handles=handles)
plt.show()