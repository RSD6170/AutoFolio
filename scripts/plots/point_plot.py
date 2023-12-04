import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from csv_reader import read_CSV


columns = ["instance", "as4mocoRun", "sbsRun", "oracleRun"]
iterations = 500

df = read_CSV(iterations, columns)
df = df.assign(instance = lambda x: x['instance'].str.extract('mc2022_track1_(\d+).dimacs'))
df["instance"] = df["instance"].astype(int)

low_cutoff = 300
df = df[(df["as4mocoRun"]>low_cutoff)&(df["sbsRun"]>low_cutoff)&(df["oracleRun"]>low_cutoff)]
#df.set_index("instance", inplace=True)

plt.scatter(x=df["instance"], y=df["as4mocoRun"], s=5, marker="*")
plt.scatter(x=df["instance"], y=df["sbsRun"], s=5, marker="x")
plt.scatter(x=df["instance"], y=df["oracleRun"], s=5, marker="v")

plt.legend(labels=["as4moco", "SBS", "oracle"])


plt.show()