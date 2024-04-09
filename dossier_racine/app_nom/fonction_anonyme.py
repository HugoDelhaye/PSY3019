import os
import pandas as pd
import numpy as np
from bin.fichier import path_load


head_path = os.path.dirname(os.path.abspath(__file__))
path = path_load(head_path)
df = pd.read_csv(path)

df_clean = df.replace(['.', 'F'], np.nan).astype(np.float64)

striatum_mean = (df_clean['r_striatum'] + df_clean['l_striatum']) / 2

carre = lambda x : x*x

striatum_carre = map(carre, striatum_mean)


