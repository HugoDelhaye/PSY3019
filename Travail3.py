import os
import pandas as pd
import numpy as np
import seaborn as sns
import warnings

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC




warnings.filterwarnings('ignore')   
warnings.simplefilter(action='ignore')

# code def implémenté
def data_load(tail_path):
    # Load the data
    head_path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(head_path, tail_path)
    data = pd.read_csv(path)
    return data

tail_path = 'HugoDelhaye_psy6973_data_20240116.csv'
df = data_load(tail_path)

# Preprossecing the data

df_clean = df.replace(['.', 'F'], np.nan).astype(np.float64)

striatum_mean = (df_clean['r_striatum'] + df_clean['l_striatum']) / 2

moca_mean = df_clean['moca'].mean()
moca_bool = df_clean['moca'] > moca_mean
moca_cat = moca_bool.astype(int)

arrig = np.asanyarray(df_clean['rigidity'])
arrtr = np.asanyarray(df_clean['tremor'])
physio = arrig + arrtr

gds_Fem = df_clean[df_clean['sex'] == 1]['gds']
gds_Mal = df_clean[df_clean['sex'] == 2]['gds']

# Combinaison des données

df_clean.insert(1, 'physio', physio, True)
df_clean.insert(1, 'striatum_mean', striatum_mean, True)
df_clean.insert(1, 'moca_cat', moca_cat, True)

# code seaborn implementé

sns.histplot(df_clean, x='striatum_mean', hue='moca_cat', kde=True)

# code préparation données

drop_out = np.where(df_clean['striatum_mean'].isna() == True)

X = np.delete(df_clean['striatum_mean'].values, drop_out, axis=0)
y = np.delete(df_clean['moca_cat'].values, drop_out, axis=0)

xtest = X.reshape(-1, 1)
ytest = y.reshape(-1, 1)

# code LDA implementé

lda = LDA(n_components=1)
X_lda = lda.fit_transform(X.reshape(-1, 1), y)

# code svm implementé

model = SVC(kernel='linear', C=1)
model.fit(X.reshape(-1, 1), y)

#code KNN implémenté

model = KNN(n_neighbors=3)
model.fit(X.reshape(-1, 1), y)

#code RandomForest implémenté

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X.reshape(-1, 1), y)

#code PCA implémenté

pca = PCA(n_components=1)
X_pca = pca.fit_transform(X.reshape(-1, 1))

