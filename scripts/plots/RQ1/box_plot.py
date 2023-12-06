import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scripts.plots import csv_reader

columns = ["instance", "as4mocoRun", "sbsRun", "oracleRun"]
iterations = 6000
solvedFrom = []
invert= False

df_init = csv_reader.read_CSV(iterations, columns)

df = csv_reader.get_solved_instances_multi(df_init, solvedFrom)
if invert:
    df = csv_reader.get_conter_instances(df_init, df)

df = df.drop("instance", axis=1)

plt.boxplot(df, labels=["as4moco", "SBS", "oracle"], meanline=True, vert=False, showmeans=True)
#plt.set_title("Split "+str(i))


plt.xlabel("Runtime [s]")

plt.show()