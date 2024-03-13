import pandas as pd

def read_CSV(iterations):

    path = "/home/ubuntu/raphael-dunkel-bachelor"
    df = pd.read_csv(get_path(path, iterations))
    df = df[df["different_solutions"]=="ALL_SAME"]
    df = df[~((df["oracleRun"]==3600) & (df["as4mocoRun"]<3600))]
    return df

def get_path(path, iterations):
    return path + "/data/Remeasurement_MCC/MCC_T1_Splits_{its}I_noFM/results.csv".format(its=iterations)

def get_solved_instances(df, columns):
    ret_value = {}
    for c in columns:
        ret_value[c] =  df[(df[c]<3600)].filter(items = ["instance", c])
    return ret_value

def get_conter_instances(df_a, df_b):
    return df_a[~df_a.instance.isin(df_b.instance)]

def get_solved_instances_multi(df, columns):
    for c in columns:
        df =  df[(df[c]<3600)]
    return df

