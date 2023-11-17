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

df["gain"] = df.apply(lambda row: ((row.sbsRun - row.as4mocoRun)/3600), axis=1)


plt.bar(df["instance"], df["gain"])

plt.show()