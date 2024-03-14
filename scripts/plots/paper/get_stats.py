import pandas as pd
import numpy as np
from scripts.plots import csv_reader

iterations = 2000

df = csv_reader.read_CSV(iterations)

solved = csv_reader.get_solved_instances(df, ["as4mocoRun", "sbsRun", "oracleRun"])

print("Successfully solved:")
print("as4moco: {}".format(solved["as4mocoRun"].shape[0]))
print("sbs: {}".format(solved["sbsRun"].shape[0]))
print("oracle: {}".format(solved["oracleRun"].shape[0]))

print("")
print(df[df['as4mocoRun']<=df['sbsRun']]['as4moco_Pipeline'].value_counts())

print("")
print("Better than instance hardness")
print(df[df['as4mocoRun']<df['instance_hardness']].count().iloc[0])

print("")
print(df[df["as4mocoRun"]<df["oracleRun"]].shape[0])
print(df[df["as4mocoRun"]<df["sbsRun"]].shape[0])

print("")
print(csv_reader.get_conter_instances(df, solved["as4mocoRun"]).to_string())