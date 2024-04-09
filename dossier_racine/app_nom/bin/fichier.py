#code
# library de fonct

import os

def path_load(head_path):
	tail_path = 'data_source/HugoDelhaye_psy6973_data_20240116.csv'
	path = os.path.join(head_path, tail_path)
	return path