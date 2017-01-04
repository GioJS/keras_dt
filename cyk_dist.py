#selezionare grammatica
#costruire CYK data frase
#convertire matrice CYK in distribuito
#applicare algoritmi 5 e 6
#verificare tramite cosine similarity l'effettiva somiglianza tra le due interpretazioni

import parserNLP.Grammar as CFG
import parserNLP.CYK as CYK
from vectors import *
from keras_dt import *
from convolutions import *

def distributed_matrix(P):
	pass


G = CFG.Grammar('S')
G.add_rules_from_file('gramm_l')
parser = CYK.CYK(G)
parser.parse('a a b')
P = parser.C
trees = parser.getTrees()
#print P
#print trees
# parser.parse('a a a b')
# P = parser.C
# trees = parser.getTrees()
# print P
# print trees
print distributed_matrix(P)