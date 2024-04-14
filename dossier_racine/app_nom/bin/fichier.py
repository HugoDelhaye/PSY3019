#code
# library de fonct

import os
import pandas as pd
import numpy as np

# au moins 5 « def » sont présent
# au moins une classe est présent

class GetDataToAnalyze:
	def __init__(self, head_path):
		self.head_path = head_path
		self.path = self.path_load()
		self.data = self.load_data()

	def path_load(self):
		tail_path = 'data_source/HugoDelhaye_psy6973_data_20240116.csv'
		path = os.path.join(self.head_path, tail_path)
		print(path)
		return path

	def load_data(self):
		
		return pd.read_csv(self.path)
	
	def clean_data(self, data):
		# code pour la gestion des erreurs est inclus
		try:
			if not isinstance(data, pd.DataFrame):
				raise TypeError('erreur: "data" doit être un DataFrame')
			# code pour vérifier la présence des valeurs manquantes implémenté
			self.data =  self.data.replace(['.', 'F'], np.nan).astype(np.float64)
			return self.data
		except TypeError as e:
			print(e)
			return None


	def data_augmentation(self, data):
		striatum_mean = (data['r_striatum'] + data['l_striatum']) / 2

		moca_mean = data['moca'].mean()
		moca_bool = data['moca'] > moca_mean
		moca_cat = moca_bool.astype(int)

		
		physio = self.fonction_recursive(np.asanyarray(data['rigidity']), 
								   		np.asanyarray(data['tremor']))

		self.data.insert(1, 'physio', physio, True)
		self.data.insert(1, 'striatum_mean', striatum_mean, True)
		self.data.insert(1, 'moca_cat', moca_cat, True)

		return self.data
	

	#code pour codage récursif, présent
	def fonction_recursive(self, list1, list2):
		if len(list1) == 0:
			return []
		else:
			return [list1[0] + list2[0]] + self.fonction_recursive(list1[1:], list2[1:])