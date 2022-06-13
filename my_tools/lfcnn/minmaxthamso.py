import pandas as pd
import csv
from itertools import zip_longest

df = pd.read_csv("datasetcbir.csv")

header = ['STT', 'min', 'max','mean ','std','var']

def min_max_scaling(df):
    # copy the dataframe
    df_norm = df.copy()

    df_col = df_norm.drop("label", axis=1)
    # apply min-max scaling
    with open("thamsothongke.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for column in df_col.columns:
            min = df_norm[column].min()
            max = df_norm[column].max()
            mean = df_norm[column].mean()
            std = df_norm[column].std()
            var = df_norm[column].var()
            writer.writerow([column,min,max,mean,std,var])
    return df_norm

min_max_scaling(df)

