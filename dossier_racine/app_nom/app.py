#Ici ce trouve mon app
# c'est le fichier principal qui va être lu
from bin.fonction_anonyme import data_load


def head_path():
	head_path = os.path.dirname(os.path.abspath(__file__))
	return head_path

df = data_load(head_path)

print(df)