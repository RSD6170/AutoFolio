import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scripts.plots import csv_reader




paths = [
     ("top", "MCC T1 full", "/home/ubuntu/raphael-dunkel-bachelor/data/after_Regressor_fix/MCC22/results_training.csv"),
     ("left", "IFM 60s", "/home/ubuntu/raphael-dunkel-bachelor/data/after_Regressor_fix/FeatureModell_Runs/60s_Extraction_Timeout/results_training.csv"),
     ("left", "IFM 30s", "/home/ubuntu/raphael-dunkel-bachelor/data/after_Regressor_fix/FeatureModell_Runs/30s_Extraction_Timeout/results_training.csv"),
     ("right", "MCC T1 F1", "/home/ubuntu/raphael-dunkel-bachelor/data/fold_runs/results_training_F1.csv"),
     ("right", "MCC T1 F2", "/home/ubuntu/raphael-dunkel-bachelor/data/fold_runs/results_training_F2.csv"),
     ("right", "MCC T1 F3", "/home/ubuntu/raphael-dunkel-bachelor/data/fold_runs/results_training_F3.csv"),
     ("right", "MCC T1 F4", "/home/ubuntu/raphael-dunkel-bachelor/data/fold_runs/results_training_F4.csv"),
     ("right", "MCC T1 F5", "/home/ubuntu/raphael-dunkel-bachelor/data/fold_runs/results_training_F5.csv")
         ]

dfs = [(i, name, pd.read_csv(path)) for i, name, path in paths]

fig, axd = plt.subplot_mosaic([['top', 'top'],['left', 'right']],
                              constrained_layout=True)



for i, name, df in dfs:
     fig.add_subplot(axd[i]).plot(df["Iterations"],df["runtime"],label=name, marker='x')

for a in fig.get_axes():
    a.legend()
    a.set_ylabel("SMAC runtime [s]")
    a.set_xlabel("SMAC Iterations")
    a.set_yscale("log")


plt.show()