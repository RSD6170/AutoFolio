import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from scripts.plots import csv_reader
import matplotlib.patches as mpatches

import matplotlib.style as style
style.use('tableau-colorblind10')


def add_identity(axes, *line_args, **line_kwargs):
    identity, = axes.plot([], [], *line_args, **line_kwargs)
    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])
    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes

iterations = 2000
cutoff = 3600

df = csv_reader.read_CSV(iterations)


df["color"] = df.apply(lambda row  : "b" if row['as4moco_Pipeline'] == "PER_SET" else "g", axis=1)

ax = plt.gca()

df_s = df[df["as4moco_Pipeline"]=="PER_SET"]
df_i = df[df["as4moco_Pipeline"]=="PER_INSTANCE"]

plt.scatter(df_s['instance_hardness'], df_s['as4mocoRun'], marker='x', color=df_s['color'], zorder=3, label="Per-Set Pipeline")
plt.scatter(df_i['instance_hardness'], df_i['as4mocoRun'], marker='+', color=df_i['color'], zorder=3, label="Per-Instance Pipeline")





ax.set_ylabel("as4moco Runtime on Instance [s] (logarithmic)")
ax.set_xlabel("Mean Solver Runtime on Instance [s] (logarithmic)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(left=0.15*10**-1)
ax.set_ylim(bottom=0.15*10**-1)
plt.hlines(3600, ax.get_xlim()[0], ax.get_xlim()[1], color="r", zorder=1)
plt.vlines(3600, ax.get_ylim()[0], ax.get_ylim()[1], color="r", zorder=2)
add_identity(axes=ax, color='lightgrey', alpha=0.8, zorder=0)
ax.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

handles, labels = ax.get_legend_handles_labels()
handles.append( Line2D([0], [0], label='Timeout', color='r'))
plt.legend(handles=handles)

#plt.show()
plt.tight_layout()
plt.savefig('../../hardness_plot_{its}I.pdf'.format(its=iterations))
