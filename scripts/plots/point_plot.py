import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


path = "/home/ubuntu/raphael-dunkel-bachelor/data/fold_runs/MCC22_T1_splits_1000I/"
before = "MCC2022_T1_F"
after = "_1000I/results.csv"
columns = ["instance", "as4mocoRun", "sbsRun", "oracleRun"]

df_arr = []
for i in range(5):
    df_i =  pd.read_csv(path+before+str(i+1)+after, usecols=columns)
    df_i["as4mocoRun"].clip(upper=3600, inplace=True)
    df_i["sbsRun"].clip(upper=3600, inplace=True)
    df_i["oracleRun"].clip(upper=3600, inplace=True)

    df_arr.append(df_i)

df = pd.concat(df_arr)
df = df.assign(instance = lambda x: x['instance'].str.extract('mc2022_track1_(\d+).dimacs'))
df["instance"] = df["instance"].astype(int)
#df.set_index("instance", inplace=True)

plt.scatter(x=df["instance"], y=df["as4mocoRun"], s=0.5)
plt.scatter(x=df["instance"], y=df["sbsRun"], s=0.5)
plt.scatter(x=df["instance"], y=df["oracleRun"], s=0.5)

plt.legend(labels=["as4moco", "SBS", "oracle"])


plt.show()