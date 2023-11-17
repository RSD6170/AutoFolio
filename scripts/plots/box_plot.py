import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fig, axs = plt.subplots(5,1, constrained_layout=True)

path = "/home/ubuntu/raphael-dunkel-bachelor/data/fold_runs/MCC22_T1_splits_1000I/"
before = "MCC2022_T1_F"
after = "_1000I/results.csv"
columns = ["as4mocoRun", "sbsRun", "oracleRun"]

for i in range(1,6):
    df = pd.read_csv(path+before+str(i)+after, usecols=columns)
    df.clip(upper=3600, inplace=True)
    axs[i-1].boxplot(df, labels=["as4moco", "SBS", "oracle"], meanline=True, vert=False)
    axs[i-1].set_title("Split "+str(i))
plt.show()