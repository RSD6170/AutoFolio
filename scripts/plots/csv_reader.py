import pandas as pd

def read_CSV(iterations, columns):

    path = "/home/ubuntu/raphael-dunkel-bachelor"

    df_arr = []
    for i in range(5):
        df_i = pd.read_csv(get_path(path, iterations, i+1), usecols=columns)
        df_i["as4mocoRun"].clip(upper=3600, inplace=True)
        df_i["sbsRun"].clip(upper=3600, inplace=True)
        df_i["oracleRun"].clip(upper=3600, inplace=True)

        df_arr.append(df_i)

    return pd.concat(df_arr)

def get_path(path, iterations, fold ):
    return path + "/data/fold_runs/MCC22_T1_splits_{its}I/MCC2022_T1_F{fold}_{its}I/results.csv".format(its=iterations, fold=fold)