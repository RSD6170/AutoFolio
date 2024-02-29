import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scripts.plots import csv_reader
import matplotlib.patches as mpatches

colors = [["lightsalmon","darkred"],["lime","seagreen"]]
timeout_color = "gold"

def colorer(row):
    if row.as4mocoRun == 3600 or row.oracleRun == 3600: return timeout_color
    fst = 1
    if row.gain >= 1: fst = 0
    snd = 1
    if row.as4moco_Pipeline == "PER_SET" : snd = 0
    return colors[fst][snd]

columns = ["instance", "as4mocoRun", "sbsRun", "oracleRun"]
iterations = 8000

df = csv_reader.read_CSV(iterations)
df = df.assign(instance = lambda x: x['instance'].str.extract('mc2022_track1_(\d+).dimacs'))
df["instance"] = df["instance"].astype(int)
#df.set_index("instance", inplace=True)

df["gain"] = df.apply(lambda row: ((row.as4mocoRun / row.sbsRun)), axis=1)
mean_log = df["gain"].apply(lambda row: math.log10(row)).mean()
mean = 10 ** mean_log
median = df["gain"].median()
df["color"] = df.apply(colorer, axis=1)
df["plot"] = df.apply(lambda row: row.gain - 1, axis=1)
print(mean)
print(median)

df_black = df[df["color"] == timeout_color]
df_whole = df[df["color"] != timeout_color]


ax = plt.gca()
plt.bar(df_whole["instance"], df_whole["plot"], color=df_whole["color"], bottom=1, log=True )
bar = plt.bar(df_black["instance"], df_black["plot"], color=df_black["color"], bottom=1, log=True)
ax.axhline(1, color='black')
ax.axhline(mean, color='b', ls='--', label="Mean")
ax.axhline(median, color='b', label="Median")


plt.ylabel("Score -- as4moco / SBS")
plt.xlabel("Identifier of input instance")

# https://stackoverflow.com/a/56551701
handles, labels = ax.get_legend_handles_labels()
handles.append( mpatches.Patch(color=colors[1][0], label='Better - Per-Set'))
handles.append( mpatches.Patch(color=colors[1][1], label='Better - Per-Instance'))
handles.append( mpatches.Patch(color=colors[0][0], label='Worse - Per-Set'))
handles.append( mpatches.Patch(color=colors[0][1], label='Worse - Per-Instance'))
handles.append( mpatches.Patch(color=timeout_color, label='Timeout'))


plt.legend(handles=handles)

plt.gca().yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

#plt.show()
plt.tight_layout()
plt.savefig('../../gain_plot_{its}I.pdf'.format(its=iterations))
