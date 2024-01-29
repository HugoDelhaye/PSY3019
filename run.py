#PSEUDO CODE:

# On importe les librairies
# On créer le DataFrame comtenant toutes les données
# On créer un DataFrame avec seulment les données qui nous intéressent 
# On tranforme ces données en array pour plus facilement les manipuler
# On créer une nouvelle colonne et on l'ajoute à l'array
# On modifie l'organisation des données dans le deuxième DataFrame grâce à la fonction groupby()

########

import pandas as pd
import numpy as np

df = pd.read_csv('HugoDelhaye_psy6973_data_20240116.csv')
print(df.columns)

# selction des colonnes d'intéret
df2 = df[['l_striatum', 'r_striatum', 'stai', 'duration', 'gds']]
df2.replace('.', np.nan)

arr = df2.to_numpy(dtype = np.float64)

# the average volum from striatum of both side
striatum_mean = ((arr[:, 0]+ arr[:, 1])/2).reshape([883, 1])

#adding a column to the initial array
arr2 = np.concatenate([arr, striatum_mean], axis=1)

by = 'gds'
df2.groupby(by)
