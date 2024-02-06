from matplotlib import pyplot as plt
import numpy
import os

# définir le chemin
chemin = os.path.join(os.environ['HOME'],'Desktop')

# visualiser les données
dat = np.array([1, 2, 4, 8, 16, 32])
plt.plot(dat, '.', color = '#000099' )
plt.show()

# sauvegarder la figure
fig = plt.figure() # pour sauvegarder, il faut créer une figure
fig.savefig(os.path.join(chemin, 'figureHugo.png'))