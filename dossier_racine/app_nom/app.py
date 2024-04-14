#Ici ce trouve mon app
# c'est le fichier principal qui va être lu
import os
from bin.fichier import Pipeline

head_path = os.path.dirname(os.path.abspath(__file__))
print(head_path)
p = Pipeline(head_path)
p.clean_data(p.data)
p.data_augmentation(p.data)


df = p.data

print(df.head())