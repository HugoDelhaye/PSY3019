from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd


fich_data = 'HugoDelhaye_psy6973_data_20240116.csv'

df = pd.read_csv(fich_data).replace('.', np.nan)
arr = df[['l_striatum', 'r_striatum']].to_numpy(dtype = np.float64)
cols = df.columns
list_na = []


#opérateurs d’agrégation présent
#boucle «for » utilisé
for i in cols:
	#valeurs manquantes cherchée
	#opérateurs de comparaison présent
	#condition « if » utilisé
    if df[i].isnull().values.any() == True:
	    list_na.append(i)

print("Les colonnes contenant des na sont : "+str(list_na))


fig, ax = plt.subplots(2)

# histogramme présent
# nuages de points présent

ax[0].hist(df['age'])
ax[0].set_xlabel('age')
ax[1].plot(arr[:, 0], arr[:, 1], '.')
ax[1].set_xlabel('Volume du striatum gauche')
ax[1].set_ylabel('Volume du striatum droit')

i =0 
#boucle « while » utilisé
while i < 3 :
	plt.show()
	plt.close()
	i = 10
