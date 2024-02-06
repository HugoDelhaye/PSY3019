from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd

# définir le chemin
chemin_main = os.path.dirname(os.path.abspath(__file__))
chemin = os.path.join(chemin_main, 'resultats','figureHugo.png')

# visualiser les données
dat = np.array([1, 2, 4, 8, 16, 32])
plt.plot(dat, '.', color = '#000099' )
plt.show()
plt.close()

# sauvegarder la figure
fig = plt.figure() # pour sauvegarder, il faut créer une figure
fig.savefig(chemin)

# importer mon cadre de données
df = pd.read_csv('HugoDelhaye_psy6973_data_20240116.csv')
df2 = df[['l_striatum']].replace('.', np.nan)
arr = df2.to_numpy(dtype = np.float64)


# visualiser les données
plt.plot(arr, 'r^')
plt.show()

# sauvegarder la figure
chemin2 = os.path.join(chemin_main, 'resultats','figureDonneesHugo.png')
fig = plt.figure()
fig.savefig(chemin2)
plt.close()

#polar 

N = arr.size
r = 2 * np.random.rand(N)
area = 200 * r**2
theta = 2 * np.pi * np.random.rand(N)

fig = plt.figure()
ax  = fig.add_subplot(111, projection='polar')
c   = ax.scatter(theta, arr, s=area, cmap='hsv', alpha=0.75)

chemin3 = os.path.join(chemin_main, 'resultats','figureDonneesHugo.png')
fig.savefig(chemin3)
plt.close()