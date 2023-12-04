import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv_reader

columns = ["instance", "as4mocoRun", "sbsRun", "oracleRun"]
iterations = 6000

df = csv_reader.read_CSV(iterations, columns)
df = df.assign(instance = lambda x: x['instance'].str.extract('mc2022_track1_(\d+).dimacs'))
df["instance"] = df["instance"].astype(int)
#df.set_index("instance", inplace=True)

df["gain"] = df.apply(lambda row: ((row.as4mocoRun / row.sbsRun) -1), axis=1)
df["color"] = df.apply(lambda row: "r" if row.gain >= 0 else "g", axis=1)
mean = df["gain"].mean()
print(mean)

ax = plt.gca()
plt.bar(df["instance"], df["gain"], bottom=1, color=df["color"])
ax.set_yscale('log')
ax.axhline(1, color='b')
ax.axhline(mean, color='b', ls='--')
yticks = [*ax.get_yticks(), mean]
yticklabels = [*ax.get_yticklabels(), round(mean,4)]
ax.set_yticks(yticks, labels=yticklabels)
ax.axhline(df["gain"].median(), color='b')



plt.show()