import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


path = "/home/ubuntu/raphael-dunkel-bachelor/data/fold_runs/MCC22_T1_splits_1000I/"
before = "MCC2022_T1_F"
after = "_1000I/results.csv"
columns = ["as4mocoRun", "sbsRun", "oracleRun"]

df_arr = []
for i in range(5):
    df_i =  pd.read_csv(path+before+str(i+1)+after, usecols=columns)
    df_i["as4mocoRun"].clip(upper=3600, inplace=True)
    df_i["sbsRun"].clip(upper=3600, inplace=True)
    df_i["oracleRun"].clip(upper=3600, inplace=True)

    df_arr.append(df_i)

df = pd.concat(df_arr)
plt.boxplot(df, labels=["as4moco", "SBS", "oracle"], meanline=True, vert=False, showmeans=True)
#plt.set_title("Split "+str(i))
plt.show()