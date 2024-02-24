''' PSEUDO CODE :

Nous importons les données sous forme de DataFrame pandas. Ensuite, nous créeon des variables à partir des
nouvelles :
- une colonne représentant la moyenne des volumes des striatum droit et gauche
- une colonne contenant les catégories : fonctions cognitives faible ou fonctions cognitives élevées
- une colonne représentant les syndromes moteurs
Puis nous utilisons groupby() et nous visualisons nos donnéès. Nous créons des graphiques pour illuster
nos hypothèse et nous effectuons les analyses.

'''

import os
import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
from scipy import stats
import matplotlib.pyplot as plt


### import data ###

fich_data = 'HugoDelhaye_psy6973_data_20240116.csv'

path2data = os.path.join(os.environ['USERPROFILE'],
                         'Desktop', 
                         'PSY3019', 
                         fich_data)

df = pd.read_csv(path2data).replace(['.', 'F'], np.nan)


### format data ###

df_clean = df.astype(np.float64)

striatum_mean = (df_clean['r_striatum'] + df_clean['l_striatum'])/2

moca_mean = df_clean['moca'].mean()

moca_cat = []

# code boucle « for
# code avec le conditionnement (if, else)
for i in df_clean['moca']:
    if i < moca_mean:
        moca_cat.append(0)
    else :
        moca_cat.append(1)

#moca_bool = df_clean['moca'] > moca_mean
#moca_cat = moca_bool.astype(int)

arrig = np.asanyarray(df_clean['rigidity'])
arrtr = np.asanyarray(df_clean['tremor'])
physio = arrtr + arrig

gds_Fem = df_clean[df_clean['sex'] == 1]['gds']
gds_Mal = df_clean[df_clean['sex'] == 2]['gds']

# combinaison des données
df_clean.insert(1, 'physio', physio, True)
df_clean.insert(1, 'striatum_mean', striatum_mean, True)
df_clean.insert(1,'moca_cat', moca_cat, True)

#application de « groupby »
df_grp = df_clean.groupby('moca_cat')
data_clean = df_clean.to_numpy(dtype = np.float64)


### visualize data ###

#nuages de pionts
# Plus le striatum est grand, moins il y a de syndromes physiologiques

x_stria = np.sort(np.random.uniform(1, 9, 500) - 0.5)
y_physio = np.flip((x_stria / 9 * 24) + (np.random.normal(0, 7, 500)))

plt.plot(x_stria, y_physio, '.')
plt.xlabel('Volume du Striatum')
plt.ylabel('Symdrômes moteurs')
plt.show()
plt.close()

# Plus les premiers syndromes se manifestent tôt, moins les syndromes moteurs sont graves

x_onset = np.sort(np.random.normal(25, 83, 500))
y_physio = (x_onset / 83 * 9 - 0.5) + ( np.random.normal(0, 34, 500))

plt.plot(x_stria, y_physio, '.')
plt.xlabel('Âge des premiers symptomes')
plt.ylabel('Symdrômes moteurs')
plt.show()
plt.close()

# Les femmes sont moins dépressives que les hommes.

gdsF  = np.random.normal(4, 1.5, 500)
gdsM  = np.random.normal(8, 1.5, 500)

plt.hist([gdsF, gdsM], label=["Gds des Femmes", "Gds des Hommes"])
plt.xlabel("Score gds")
plt.ylabel("Fréquence")
plt.legend()
plt.show()

### analyses ###

model1 = ols("physio ~ ageonset", df_clean).fit()
model2 = ols("physio ~ striatum_mean", df_clean).fit()
testt3 = stats.ttest_ind(gds_Fem, gds_Mal)

print(model1.summary())
print(model2.summary())
print(testt3.statistic, testt3.pvalue)

