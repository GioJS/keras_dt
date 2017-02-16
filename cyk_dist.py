from parserNLP.Grammar import Grammar
from parserNLP.CYK import CYK
from vectors import *
#from keras_dt import *
from convolutions import *

dim = 1024*2
#dt = DT(dim=1024, lexicalized=True)
gen = Vector_generator(dim=dim)
Phi = permutation_matrices(dim)[1]
v = gen.get_random_vector
#print Phi

#[v]+
def sc(v):
    return circulant(v).dot(Phi)
#[v]-
def invsc(v):
    return sc(v).T
index0 = sc(v('0'))
index1 = sc(v('1'))
index2 = sc(v('2'))
index3 = sc(v('3'))
D = sc(v('D'))
E = sc(v('E'))
S = sc(v('S'))
sep = sc(v('Sep'))

def init_simple(w):
    P = np.array([0])
    for i in range(len(w)):
        s = index0.dot(sc(v(str(i+1)))).dot(sc(v(w[i])))
        P = P + s
    return P
#perterminal rules
def preterminals_simple(P,G,w):
    R = np.array([0])

    for rule in G.get_unit_productions():
        #print rule
        R = R + sc(v(rule.head())).dot(invsc(v(rule.production())))

    s = np.array([0])
    for i in range(len(w)):
        s = s + index1.dot(sc(v(str(i+1)))).dot(R).dot(invsc(v(str(i+1)))).dot(index0.T).dot(P)
    P = s
    return P



def compute_R(G, rules_A):
    Ra = np.zeros((dim,dim))
    for i in rules_A:
        rule = G[i]
        if not rule.is_preterminal():
            #print rule
            Ra = Ra + sc(v(rule[0])).dot(invsc(v(rule[1])))
    return Ra

def binary_simple(P,G,w):
    n = len(w)
    #C = compute_C(G)
    #s = np.array([0])
    R = {}
    #print 'preterminal: \n',P
    # G.groups = non-terminals
    for A in G.groups:
        rules_A = G.get_rules(A)
        R[A] = compute_R(G, rules_A)
    for i in range(2,n):
        for j in range(1,n-i+2):
            #print i,j
            for A in G.groups:
                Ra = R[A]
                #print 'A: ',A,'Ra: \n',Ra
                Pa = np.array([0])
                for k in range(1,i):
                    #print "j,k:",j,k
                    #print "j+k,i-k",j+k,i-k
                    Pa = Pa + invsc(v(str(j))).dot(invsc(v(str(k)))).dot(P).dot(Ra).dot(invsc(v(str(j+k)))).dot(invsc(v(str(i-k)))).dot(P)
                #print 'Pa:\n',Pa
                #print (Pa==0).all()
                s = sc(v(str(i))).dot(sc(v(str(j)))).dot(sc(v(A))).dot(Pa)
                #print 's: \n',s
                P = P + s
                #print 'P new: \n',P
    #P = P + s
    return P
#transform P to P_dist with algo5,6
def cyk_dist_simple(G,w):
    w = w.replace(' ','')
    P_dist = init_simple(w)
    #print P_dist
    P_dist = preterminals_simple(P_dist, G, w)
    #print P_dist
    P_dist = binary_simple(P_dist, G, w)
    return P_dist

#initialization of level 0
def init(w):
    P = np.array([0])
    for i in range(len(w)):
        s = index0.dot(sc(v(str(i+1)))).dot(sc(v(w[i]))).dot(sep)
        P = P + s
    return P
#perterminal rules
def preterminals(P,G,w):
    R = np.array([0])

    for rule in G.get_unit_productions():
        #print rule
        R = R + (sc(v(rule.head())).dot(sep).dot(sc(v(rule.head()))).dot(sep).dot(sc(v(rule.production()))).dot(sep.T).dot(invsc(v(rule.head())))).dot(sep.T).dot(invsc(v(rule.production())))

    s = np.array([0])
    for i in range(len(w)):
        s = s + index1.dot(sc(v(str(i+1)))).dot(R).dot(invsc(v(str(i+1)))).dot(index0.T).dot(P)
    P = s
    return P


def compute_C(G):
    C = np.array([0])
    for A in G.groups:
        C = C + sc(v(A)).dot(sep)
    return C
def compute_R(G, rules_A):
    Ra = np.zeros((dim,dim))
    for i in rules_A:
        rule = G[i]
        if not rule.is_preterminal():
            #print rule
            Ra = Ra + sc(v(rule[0])).dot(sep).dot(Phi).dot(sep.T).dot(invsc(v(rule[1])))
    return Ra

def binary(P,G,w):
    n = len(w)
    C = compute_C(G)
    #s = np.array([0])
    R = {}
    print 'preterminal: \n',P
    # G.groups = non-terminals
    for A in G.groups:
        rules_A = G.get_rules(A)
        R[A] = compute_R(G, rules_A)
    for i in range(2,n):
        for j in range(1,n-i+2):
            #print i,j
            for A in G.groups:
                Ra = R[A]
                #print 'A: ',A,'Ra: \n',Ra
                Pa = np.array([0])
                for k in range(1,i):
                    #print "j,k:",j,k
                    #print "j+k,i-k",j+k,i-k
                    Pa = Pa + C.dot((invsc(v(str(j))))).dot(invsc(v(str(k)))).dot(P).dot(Ra).dot(invsc(v(str(j+k)))).dot(invsc(v(str(i-k)))).dot(P).dot(C.T)
                #print 'Pa:\n',Pa
                #print (Pa==0).all()
                s = sc(v(str(i))).dot(sc(v(str(j)))).dot(sc(v(A))).dot(sep).dot(sc(v(A))).dot(sep).dot(Pa).dot(sep.T).dot(invsc(v(A)))
                #print 's: \n',s
                P = P + s
                #print 'P new: \n',P
    #P = P + s
    return P
#transform P to P_dist with algo5,6
def cyk_dist(G,w):
	w = w.replace(' ','')
	P_dist = init(w)
	#print P_dist
	P_dist = preterminals(P_dist, G, w)
	#print P_dist
	P_dist = binary(P_dist, G, w)
	return P_dist


#trasformazione di P in distributed with trees
def tree_dist(t):
    if len(t) == 0:
        return sc(v(t.label))
    s = sc(v(t.label)).dot(sep)
    for child in t:
        s = s.dot(tree_dist(child))
    return s

def test_P(parser,w):
    w = w.replace(' ','')
    Dp = np.array([0])
    #test

    for i in range(len(w)):
        s = index0.dot(sc(v(str(i+1)))).dot(sc(v(w[i])))
        Dp = Dp + s

    #first row
    for i in range(len(parser.C)):
        #preterminal trees
        #print chart
        for A in parser.C[i,i]:
            tree = parser.get_tree(A)
            #print 'tree: ',tree
            td = index1.dot(sc(v(str(i+1)))).dot(tree_dist(tree))
            Dp = Dp + td
    #generic row
    for i in range(2,len(w)):
        for j in range(0,len(w)-i+2):
            if i==j:
                continue
            for A in parser.C[j,i]:
                #print A
                #print i,j
                tree = parser.get_tree(A)
                #print 'tree: ',tree
                #P is 3x3 Dp is 4x3
                #print i+1-j, j+1, A
                td = sc(v(str(i+1-j))).dot(sc(v(str(j+1)))).dot(tree_dist(tree))
                #print td
                Dp = Dp + td
    return Dp


G = Grammar('S')
G.add_rules_from_file('gramm_l')
parser = CYK(G)
for i in range(2,3):
    w = ('a '*i)+'b'
    #w = 'a'
    print w
    #parser.parse(w)
    #P = parser.C
    #print P
    from trees import *
    dist_P = cyk_dist_simple(G,w)
    print dist_P
    rule = index2.dot(index2).dot(S)
    print rule
    print dist_P.dot(rule)
    #P_1 = init(w)
    #P_1 = preterminals(P_1,G,w)
    #Pa = Pa + C.dot((invsc(v(str(j))))).dot(invsc(v(str(k)))).dot(P).dot(Ra).dot(invsc(v(str(j+k)))).dot(invsc(v(str(i-k)))).dot(P).dot(C.T)
    #C = compute_C(G)
    '''P_11 = index1.dot(index1).dot(D).dot(sep).dot(tree_dist(Tree.from_penn('(D a)'))).dot(sep.T).dot(D.T)
    P_12 = index1.dot(index2).dot(D).dot(sep).dot(tree_dist(Tree.from_penn('(D a)'))).dot(sep.T).dot(D.T)
    P_13 = index1.dot(index3).dot(E).dot(sep).dot(tree_dist(Tree.from_penn('(E b)'))).dot(sep.T).dot(E.T)'''

    #rule = D.dot(sep).dot(Phi).dot(sep.T).dot(E.T)
    #fin = S.dot(sep).dot(sep.T).dot(D.T).dot(index2.T).dot(index1.T).dot(P_12).dot(rule).dot(index3.T).dot(index1.T).dot(P_13).dot(E).dot(sep)
    #t = tree_dist(Tree.from_penn('(S (D a) (E b))'))
    #t1 = S.dot(sep).dot(tree_dist(Tree.from_penn('(D a)'))).dot(Phi).dot(tree_dist(Tree.from_penn('(E b)')))
    #t2 = S.dot(sep).dot(sep.T).dot(D.T).dot(D).dot(sep).dot(tree_dist(Tree.from_penn('(D a)'))).dot(sep.T).dot(D.T).dot(D).dot(sep).dot(Phi).dot(sep.T).dot(E.T).dot(index3.T).dot(index1.T).dot(index1).dot(index3).dot(E).dot(sep).dot(tree_dist(Tree.from_penn('(E b)'))).dot(sep.T).dot(E.T).dot(E).dot(sep)
    #print t1.dot(t2.T)
    #dot(D.T).dot(index2.T).dot(index1.T).dot(index1).dot(index2).dot(D)
    #Pd = cyk_dist(G,w)
    #print Pd.dot(P_1.T)
    #print sep.dot(sep.T)
    #print sep.T.dot(D.T).dot(sep.T).dot(sep).dot(D).dot(sep)
    #print t1.dot(t2.T)
    #x = circulant(v('sdfg'))
    '''for i in range(100):
        Pd = Pd + circulant(v('Prova'+str(i)))'''
    #Pd = invsc(v('1')).dot(invsc(v('2'))).dot(Pd)
    #Pd = invsc(v("1")).dot(invsc(v("0"))).dot(Pd).dot(invsc(v('Sep')))

    #Pd = invsc(v('Sep')).dot(invsc(v('S'))).dot(Pd)
    #Pd = Pd.dot(sc(v('S'))).dot(sc(v('Sep')))
    '''C = compute_C(G)
    print C.dot(C.T)
    print C.T.dot(C)'''
    #Pd = C.T.dot(Pd).dot(C)
    #Pd = sc(v('S')).dot(Pd) #test
    #Dw0 = sc(v('D')).dot(circulant(v('a')))
    #t_d = tree_dist(Tree('D',[Tree('a',[])]))
    #from parserNLP.Rule import *
    #tentando di costruire a mano l'elemento secondo binary
    '''C = compute_C(G)
    rules_A = G.get_rules('S')
    Ra = compute_R(G, rules_A)
    Pa = C.dot(D).dot(sep).dot(tree_dist(Tree.from_penn('(D a)'))).dot(sep.T).dot(D.T).dot(Ra).dot(E).dot(sep).dot(tree_dist(Tree.from_penn('(E b)'))).dot(sep.T).dot(E.T).dot(C.T)
    el = index2.dot(index1).dot(S).dot(sep).dot(S).dot(sep).dot(Pa).dot(sep.T).dot(S.T)'''
    #rule = Rule('D','a',0)
    #term = sc(v('0')).dot(sc(v('1'))).dot(sc(v('a'))).dot(sc(v('Sep')))
    #print Pd.dot(term.T)
    #print tree_dist(Tree.from_penn('(D a)')).dot(invsc(v(rule.production()))).dot(invsc(v('Sep'))).dot(invsc(v(rule.head())))
    #t_d = index1.dot(index1).dot(D).dot(sep).dot(tree_dist(Tree.from_penn('(D a)'))).dot(sep.T).dot(D.T)
    #print Pd.dot(el.T)
    #t_d = tree_dist(Tree.from_penn('(S (D a) (E b))'))
    #t_d = tree_dist(Tree.from_penn('(S (D a) (S (D a) (E b)))'))
    #print t_d[:,0].dot(Dw0[:,0])
    #print Pd.dot(t_d.T)
    