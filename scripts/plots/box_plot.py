import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from csv_reader import read_CSV


columns = ["as4mocoRun", "sbsRun", "oracleRun"]
iterations = 500

df = read_CSV(iterations, columns)
plt.boxplot(df, labels=["as4moco", "SBS", "oracle"], meanline=True, vert=False, showmeans=True)
#plt.set_title("Split "+str(i))
plt.show()