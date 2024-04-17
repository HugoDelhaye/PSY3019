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
from sklearn.model_selection import cross_val_score
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
sns.regplot(df, x='striatum_mean', y='physio', marker = "+")
plt.show()
plt.close()

# Plus les premiers syndromes se manifestent tôt, moins les syndromes moteurs sont graves
sns.regplot(df, x='ageonset', y='physio', marker = "+")
plt.show()
plt.close()

####### entrainement des modèles

# code préparation données

drop_out = np.where(df['striatum_mean'].isna() == True)

X_AA_sup = np.delete(df['striatum_mean'].values, drop_out, axis=0)
y_AA_sup = np.delete(df['moca_cat'].values, drop_out, axis=0)

X_AA_sup = np.ravel(X_AA_sup)
y_AA_sup = np.ravel(y_AA_sup)

# sépare gds du reste
X_AA_non_sup = df.drop('gds', axis=1)
y_AA_non_sup = df.gds

drop_out = np.where(df['striatum_mean'].isna() == True)

X_AA_non_sup = np.delete(df['striatum_mean'].values, drop_out, axis=0)
y_AA_non_sup = np.delete(df['moca_cat'].values, drop_out, axis=0)

X_AA_non_sup = np.ravel(X_AA_sup).reshape(-1, 1)
y_AA_non_sup = np.ravel(y_AA_sup)


# entrainment des modeles
model1 = SVC(kernel='linear', C=1)

#code pour la validation croisée est utilisé
scores1 = cross_val_score(model1, X_AA_sup.reshape(-1, 1), y_AA_sup, cv=10)

# Fit the model
model1.fit(X_AA_sup.reshape(-1, 1), y_AA_sup)

# Make predictions
predictions1 = model1.predict(X_AA_sup.reshape(-1, 1))

# Create confusion matrix
confusion_matrix1 = confusion_matrix(y_AA_sup, predictions1)
sns.heatmap(confusion_matrix1.T, square=True, annot=True, fmt='d', cbar=False)
plt.show()
plt.close()

# code pour le formatage des chaînes est utilisé
print("Accuracy: %0.2f (+/- %0.2f)" % (scores1.mean(), scores1.std() * 2))

model2 = KMeans(n_clusters=10)

#code pour la validation croisée est utilisé
scores2 = cross_val_score(model2, X_AA_non_sup, y_AA_non_sup, cv=10)

# Fit the model
model2.fit(X_AA_non_sup.reshape(-1, 1), y_AA_non_sup)

# Make predictions
predictions2 = model2.predict(X_AA_non_sup.reshape(-1, 1))

# Create confusion matrix
confusion_matrix2 = confusion_matrix(y_AA_non_sup, predictions2)
sns.heatmap(confusion_matrix2.T, square=True, annot=True, fmt='d', cbar=False)
plt.show()
plt.close()
