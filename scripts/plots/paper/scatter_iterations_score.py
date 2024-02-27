import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scripts.plots import csv_reader




iterations = [500, 1000, 2000, 4000, 6000, 8000]

scores = []

for iteration in iterations:
    df = csv_reader.read_CSV(iteration)

    for fold in range(1,6):
        dff = df[df['Fold'] == fold]
        as4moco = dff["as4mocoRun"].mean()
        sbs = dff["sbsRun"].mean()
        oracle = dff["oracleRun"].mean()
        score = (as4moco - oracle) / (sbs - oracle)
        scores.append({"iteration": iteration, "fold": "MCC22 T1 F{fold}".format(fold=fold), "score": score})

    as4moco = df["as4mocoRun"].mean()
    sbs = df["sbsRun"].mean()
    oracle = df["oracleRun"].mean()
    score = (as4moco - oracle) / (sbs - oracle)
    scores.append({"iteration": iteration, "fold": "MCC22 T1 complete", "score": score})

df = pd.DataFrame(scores)
for name, groups in df.groupby('fold'):
    plt.plot(groups['iteration'], groups['score'], marker='x', label=name)

a = plt.gca()
a.legend()
a.set_ylabel("Closed Gap Score")
a.set_xlabel("SMAC Iterations")
#a.set_yscale("log")
a.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
a.set_ylim(bottom=0)

plt.show()
