import pandas as pd
import numpy as np

df = pd.read_csv("data/uncleaned.csv")
# df_2 = pd.read_csv("data/switzerland_data.csv")
# df_3 = pd.read_csv("data/hungarian_data.csv")
# df_4 = pd.read_csv("data/cleveland_data.csv")

df = df.drop(columns=["trestbps", "chol", "fbs", "restecg"])

dt_clean = df.dropna(axis=0, how='any').reset_index().drop(columns=['index'])

dt_new = pd.DataFrame(columns=dt_clean.columns.tolist())

for index in range(len(dt_clean)):
    num = dt_clean.at[index, 'num']
    if num>1:
        dt_clean.at[index, 'num'] = 1
        row = dt_clean.iloc[index]
        row = row.to_dict()
        for i in range(num-1):
            dt_new = dt_new.append(row, ignore_index=True)

dt_clean = pd.DataFrame(pd.concat([dt_clean, dt_new], axis=0, ignore_index=True)).reset_index().drop(columns=['index'])

dt_clean.sample(frac=1)

dt_clean.to_csv("data/cleaned.csv")
