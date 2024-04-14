#Ici ce trouve mon app
# c'est le fichier principal qui va être lu
import os
import pandas as pd
import numpy as np
import seaborn as sns
from statsmodels.formula.api import ols
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from bin.fichier import GetDataToAnalyze



head_path = os.path.dirname(os.path.abspath(__file__))
print(head_path)
p = GetDataToAnalyze(head_path)
p.clean_data(p.data)
p.data_augmentation(p.data)
df = p.data

gdsF = df[df['sex'] == 1]['gds']
gdsM = df[df['sex'] == 2]['gds']

# code pour stats utilise les modules statsmodels + SciPy

model1 = ols("physio ~ ageonset", df).fit()
model2 = ols("physio ~ striatum_mean", df).fit()
testt3 = stats.ttest_ind(gdsF, gdsM)

print(model1.summary())
print(model2.summary())
print(testt3.statistic, testt3.pvalue)

# graphique pour les stats (statsmodels, SciPy) présent

# Les femmes sont moins dépressives que les hommes.
plt.hist([gdsF, gdsM], label=["Gds des Femmes", "Gds des Hommes"])
plt.xlabel("Score gds")
plt.ylabel("Fréquence")
plt.legend()
plt.show()
plt.close()

# Plus le striatum est grand, moins il y a de syndromes physiologiques
sns.regplot(df, x='striatum_mean', hue='physio', marker = "+")
plt.show()
plt.close()

# Plus les premiers syndromes se manifestent tôt, moins les syndromes moteurs sont graves
sns.regplot(df, x='ageonset', hue='physio', marker = "+")
plt.show()
plt.close()

####### entrainement des modèles

# code préparation données

drop_out = np.where(df['striatum_mean'].isna() == True)

X_AA_non_sup = np.delete(df['striatum_mean'].values, drop_out, axis=0)
y_AA_non_sup = np.delete(df['moca_cat'].values, drop_out, axis=0)

xtest = X_AA_non_sup.reshape(-1, 1)
ytest = y_AA_non_sup.reshape(-1, 1)

# sépare le scrore de depression du reste
X_AA_sup = df.drop('', axis=1)
X_AA_sup = df.

# code svm implementé

model = SVC(kernel='linear', C=1)
model.fit(X_AA_non_sup.reshape(-1, 1), y_AA_non_sup)

# cross validation du svm 


# code k-moyennes implementé

model = KMeans(n_clusters=2)
model.fit(X.reshape(-1, 1))
