import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv_reader

columns = ["instance", "as4mocoRun", "sbsRun", "oracleRun"]
iterations = 500

df = csv_reader.read_CSV(iterations, columns)
df = df.assign(instance = lambda x: x['instance'].str.extract('mc2022_track1_(\d+).dimacs'))
df["instance"] = df["instance"].astype(int)
#df.set_index("instance", inplace=True)

df["gain"] = df.apply(lambda row: ((row.as4mocoRun / row.sbsRun) -1), axis=1)

ax = plt.gca()
plt.bar(df["instance"], df["gain"], bottom=1)
ax.set_yscale('log')
plt.hlines(1, 0, 200, 'r')

plt.show()