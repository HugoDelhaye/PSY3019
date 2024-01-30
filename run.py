#PSEUDO CODE:

'''

On importe des données d'interet sous forme de DataFrame pandas et de Array numpy. Puis nous créons une nouvelle
colonne représentant la moyenne des volumes des striatum droit et gauche pour chaque participants, cette colonne
est ajoutée dans l'Array. Pour finir nous reformatons le DataFrame grâce à groupby().

# On modifie l'organisation des données dans le deuxième DataFrame grâce à la fonction groupby()

'''

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
