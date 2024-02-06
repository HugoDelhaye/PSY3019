from matplotlib import pyplot as plt
dat = np.array([1, 2, 4, 8, 16, 32])
plt.plot(dat, '.', color = '#000099' )
plt.show

fig = plt.figure() # pour sauvegarder, il faut créer une figure
fig.savefig(os.path.join(chemin, 'my_figure.png'))