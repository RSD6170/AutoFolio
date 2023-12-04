import pandas as pd
import numpy as np
import csv_reader

columns = ["instance", "as4mocoRun", "sbsRun", "oracleRun", "sbs-oracle","sbs-as4moco","as4moco-oracle"]
iterations = 500

df = csv_reader.read_CSV(iterations, columns)

solved = csv_reader.get_solved_instances(df, ["as4mocoRun", "sbsRun", "oracleRun"])

print("Successfully solved:")
print("as4moco: {}".format(solved["as4mocoRun"].shape[0]))
print("sbs: {}".format(solved["sbsRun"].shape[0]))
print("oracle: {}".format(solved["oracleRun"].shape[0]))

print("")

print("Same solver:")
print(df["sbs-oracle"].value_counts())
print(df["sbs-as4moco"].value_counts())
print(df["as4moco-oracle"].value_counts())

print("")
print(csv_reader.get_conter_instances(df, solved["as4mocoRun"]))