import pandas as pd
import numpy as np


df = pd.read_csv('HugoDelhaye_psy6973_data_20240116.csv')
print(df)
cols = df.columns
df2 = df.replace('.', np.nan)
print(df2)

list_na = []

for i in cols:
	if df2[i].isnull().values.any() == True:
		list_na.append(i)

print("Les colonnes contenant des na sont : "+str(list_na))