from nltk import CFG, ChartParser
from random import choice
from cyk_dist import *
import numpy as np

def produce(grammar, symbol, w=[]):
    words = []
    productions = grammar.productions(lhs = symbol)
    production = choice(productions)
    for sym in production.rhs():
        if isinstance(sym, str):
            words.append(sym)
        else:
            words.extend(produce(grammar, sym))
    if ' '.join(words) in w:
        return produce(grammar, symbol, w)
    return words

grammar = CFG.fromstring('''
S -> D S
S -> D E
D -> 'a'
E -> 'b'
''')

parser = ChartParser(grammar)
G = Grammar('S')
G.add_rules_from_file('gramm_l')
gr = parser.grammar()
w_old = []
for i in range(10):
    w = ' '.join(produce(gr, gr.start(),w_old))
    w_old.append(w)
    print(w)
    P = init_simple(w)
    P = preterminals_simple_with_sigmoid(P,G,w)
    np.save("testset/{}.npy".format(w),P,allow_pickle=True)
