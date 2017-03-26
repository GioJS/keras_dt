from nltk import CFG, ChartParser
from random import choice
import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cyk_dist import *





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

with open('gramm_m_nltk') as f:
    grammar = CFG.fromstring(''.join(f.readlines()))

parser = ChartParser(grammar)
G = Grammar('S')
parserG = CYK(G)
G.add_rules_from_file('gramm_m')
gr = parser.grammar()
w_old = []
for i in range(10):
    w = ' '.join(produce(gr, gr.start(),w_old))
    w_old.append(w)
    print(w)
    P = cyk_dist_simple(G, w)
    parserG.parse(w)
    '''with open("testset/{}.cyk".format(w.replace(' ','')), 'w') as f:
        f.write(str(parserG.C))'''
    np.savetxt("{}.cyk".format(w),parserG.C,newline='\r\n',fmt="%s", delimiter=' ')
    np.save("{}_{}.npy".format(w,dim),P, allow_pickle=True)