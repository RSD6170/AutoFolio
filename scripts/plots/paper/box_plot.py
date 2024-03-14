import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scripts.plots import csv_reader

def get_box_plot_data(labels, bp):
    rows_list = []

    for i in range(len(labels)):
        dict1 = {}
        dict1['label'] = labels[i]
        dict1['lower_whisker'] = bp['whiskers'][i*2].get_xdata()[1]
        dict1['lower_quartile'] = bp['boxes'][i].get_xdata()[1]
        dict1['median'] = bp['medians'][i].get_xdata()[1]
        dict1['upper_quartile'] = bp['boxes'][i].get_xdata()[2]
        dict1['upper_whisker'] = bp['whiskers'][(i*2)+1].get_xdata()[1]
        rows_list.append(dict1)

    return pd.DataFrame(rows_list)

iterations = 2000
solvedFrom = []
invert= False

df_init = csv_reader.read_CSV(iterations)

df = csv_reader.get_solved_instances_multi(df_init, solvedFrom)
if invert:
    df = csv_reader.get_conter_instances(df_init, df)


test = plt.boxplot(df[["as4mocoRun", "sbsRun", "oracleRun"]], labels=["as4moco", "SBS", "Oracle"], meanline=True, vert=False, showmeans=True)
#plt.set_title("Split "+str(i))

xMinus = 0
yMinus = 0.45

print("Median")
for line in test['medians']:
    x, y = line.get_xydata()[1]
    plt.annotate(f"{x:.2f}", xy=(x-xMinus, y-yMinus), horizontalalignment='center')
    print(x, y)

print("Means")
for line in test['means']:
    x, y = line.get_xydata()[1]
    plt.annotate(f"{x:.2f}", xy=(x-xMinus, y-yMinus), horizontalalignment='center')
    print(x, y)

print("caps")
for line in test['caps']:
    x, y = line.get_xydata()[1]
    print(x, y)

print(get_box_plot_data(["as4moco", "SBS", "Oracle"], test).to_string())

plt.legend([test['medians'][0], test['means'][0]], ['Median', 'Mean'])
plt.gca().xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

plt.xlabel("Runtime [s]")

#plt.show()
plt.tight_layout()
plt.savefig('../../box_plot_{its}I.pdf'.format(its=iterations))