import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scripts.plots import csv_reader

import matplotlib.style as style
style.use('tableau-colorblind10')



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


df = pd.DataFrame(scores)

df_mod = df.drop('fold', axis=1).groupby('iteration')
groups = [group['score'].tolist() for _, group in df_mod]
labels = [f"{key}" for key, _ in df_mod]
test = plt.boxplot(groups, labels=labels, meanline=True, vert=True, showmeans=True, meanprops=dict(linestyle='-'))

xMinus = 0.1
yMinus = 0



for line in test['medians']:
    x, y = line.get_xydata()[1]
    #plt.annotate(f"{y:.2f}", xy=(x-xMinus, y-yMinus), verticalalignment='top', horizontalalignment='center')

for line in test['means']:
    x, y = line.get_xydata()[1]
    #plt.annotate(f"{y:.2f}", xy=(x-xMinus, y-yMinus), verticalalignment='bottom', horizontalalignment='center')

X = []
Y = []
for m in test['medians']:
    [[x0, x1], [y0, y1]] = m.get_data()
    X.append(np.mean((x0, x1)))
    Y.append(np.mean((y0, y1)))
plt.plot(X, Y, c='C1', marker='o', label='Median')
X = []
Y = []
for m in test['means']:
    [[x0, x1], [y0, y1]] = m.get_data()
    X.append(np.mean((x0, x1)))
    Y.append(np.mean((y0, y1)))
plt.plot(X, Y, c='C2', marker='o', label='Mean')

plt.legend()
plt.gca().yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

plt.xlabel("SMAC Iterations")
plt.ylabel("Closed Gap Score")

#plt.show()
plt.tight_layout()
plt.savefig('../../box_it_score_plot.pdf')

