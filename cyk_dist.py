#selezionare grammatica
#costruire CYK data frase
#convertire matrice CYK in distribuito
#applicare algoritmi 5 e 6
#verificare tramite cosine similarity l'effettiva somiglianza tra le due interpretazioni

import cyk.Grammar as CFG
import cyk.CYK as CYK

G = CFG.Grammar('S')
G.add_rules_from_file('gramm_l')
parser = CYK.CYK(G, 'a a b')
parser.parse()
P = parser.C
trees = parser.getTrees()
print P
print trees