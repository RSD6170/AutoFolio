import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scripts.plots import csv_reader
import matplotlib.patches as mpatches


iterations = 8000
cutoff = 3600

df = csv_reader.read_CSV(iterations)


df["color"] = df.apply(lambda row  : "b" if row['as4moco_Pipeline'] == "PER_SET" else "g", axis=1)

ax = plt.gca()

plt.scatter(df['instance_hardness'], df['as4mocoRun'], marker='x', color=df['color'])


handles, labels = ax.get_legend_handles_labels()
better = mpatches.Patch(color='b', label='Per-Set Pipeline')
worse = mpatches.Patch(color='g', label='Per-Instance Pipeline')

handles.append(better)
handles.append(worse)

plt.legend(handles=handles)

ax.set_ylabel("as4moco Runtime on Instance [s] (logarithmic)")
ax.set_xlabel("Mean Solver Runtime on Instance [s] (logarithmic)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

#plt.show()
plt.tight_layout()
plt.savefig('../../hardness_plot_{its}I.pdf'.format(its=iterations))
