from parserNLP.Grammar import Grammar
from parserNLP.CYK import CYK
from vectors import *
#from keras_dt import *
from convolutions import *

dim = 1024
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

#initialization of level 0
def init(w):
    P = np.array([0])
    for i in range(len(w)):
        s = (sc(v('0')).dot(sc(v(str(i+1)))).dot(sc(v(w[i]))).dot(sc(v('Sep'))))
        P = P + s
    return P
#perterminal rules
def preterminals(P,G,w):
    R = np.array([0])

    for rule in G.get_unit_productions():
        R = R + (sc(v(rule.head())).dot(sc(v('Sep'))).dot(sc(v(rule.head()))).dot(sc(v(rule.production()))).dot(invsc(v('Sep'))).dot(invsc(v(rule.head())))).dot(invsc(v('Sep'))).dot(invsc(v(rule.production())))

    s = np.array([0])
    for i in range(len(w)):
        s = s + (sc(v('1')).dot(sc(v(str(i+1)))).dot(R).dot(invsc(v(str(i+1)))).dot(invsc(v('0'))).dot(P))
    P = P + s
    return P


def compute_C(G):
    C = np.array([0])
    for A in G.groups:
        C = C + sc(v(A)).dot(sc(v('Sep')))
    return C
def compute_R(G, rules_A):
    Ra = np.zeros((dim,dim))
    for i in rules_A:
        rule = G[i]
        if not rule.is_preterminal():
            #print rule
            Ra = Ra + sc(v(rule[0])).dot(sc(v('Sep'))).dot(Phi).dot(invsc(v('Sep'))).dot(invsc(v(rule[1])))
    return Ra

def binary(P,G,w):
    n = len(w)
    C = compute_C(G)
    #s = np.array([0])
    R = {}
    # G.groups = non-terminals
    for A in G.groups:
        rules_A = G.get_rules(A)
        R[A] = compute_R(G, rules_A)
    for i in range(2,n+1):
        for j in range(1,n-i+1):
            for A in G.groups:
                Ra = R[A]
                Pa = np.array([0])
                for k in range(1,i+2):
                    Pa = Pa + C.dot((invsc(v(str(j))))).dot(invsc(v(str(k)))).dot(P).dot(Ra).dot(invsc(v(str(j+k)))).dot(invsc(v(str(i-k)))).dot(P).dot(C.T)
                #print (Pa==0).all()
                P = P + sc(v(str(i))).dot(sc(v(str(j)))).dot(sc(v(A))).dot(sc(v('Sep'))).dot(sc(v(A))).dot(sc(v('Sep'))).dot(Pa).dot(invsc(v('Sep'))).dot(invsc(v(A)))
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
    s = sc(v(t.label)).dot(sc(v('Sep')))
    for child in t:
        s = s.dot(tree_dist(child))
    return s

def test_P(parser,w):
    w = w.replace(' ','')
    Dp = np.array([0])
    #test

    for i in range(len(w)):
        s = sc(v('0')).dot(sc(v(str(i+1)))).dot(sc(v(w[i])))
        Dp = Dp + s

    #first row
    for i in range(len(parser.C)):
        #preterminal trees
        #print chart
        for A in parser.C[i,i]:
            tree = parser.get_tree(A)
            #print 'tree: ',tree
            td = sc(v("1")).dot(sc(v(str(i+1)))).dot(tree_dist(tree))
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
    parser.parse(w)
    P = parser.C
    print P
    #print parser.get_tree(P[2,2][0])
    #if K.backend() == 'tensorflow':
        # sess = K.tf.Session()
        # K.set_session(sess)
        # with sess.as_default():
        #     '''a = sc(v('0')).dot(sc(v('0'))).dot(sc(v('a'))).dot(sc(v('Sep')))
        #     a_0 = invsc(v('0')).dot(a)
        #     a_1 = sc(v('0')).dot(sc(v('a'))).dot(sc(v('Sep')))
        #     print a_0
        #     print a_1
        #     print np.linalg.norm(a_0-a_1,2)'''

    	   #  a = v('a')
    	   #  A = circulant(a)
    	   #  print A.dot(A.T)
            #Pd = cyk_dist(G,w)
            #print sc(v("0")),sc(v("0"))
            #print np.linalg.norm(sc(v("Sep"))-invsc(v("Sep")),2)
            #Pd = invsc(v('0')).dot(invsc(v('0'))).dot(Pd)
            #Pd = Pd.dot(invsc(v('Sep')))
            #print Pd
            #Dp = test_P(parser,w)
            #Dp = invsc(v('0')).dot(invsc(v('0'))).dot(Dp)
            #print Dp
            #print np.linalg.norm(Pd-Dp,2)
            #print invsc(v("0")).dot(invsc(v("1"))).dot(Pd)
            #print invsc(v("0")).dot(invsc(v("1"))).dot(Dp)
    #else:


    Pd = cyk_dist(G,w)
    Pd = invsc(v('2')).dot(invsc(v('2'))).dot(Pd)
    #Pd = invsc(v("1")).dot(invsc(v("0"))).dot(Pd).dot(invsc(v('Sep')))

    Pd = invsc(v('Sep')).dot(invsc(v('S'))).dot(Pd)
    Pd = Pd.dot(sc(v('S'))).dot(sc(v('Sep')))
    '''C = compute_C(G)
    print C.dot(C.T)
    print C.T.dot(C)'''
    #Pd = C.T.dot(Pd).dot(C)
    #Pd = sc(v('S')).dot(Pd) #test
    #Dw0 = sc(v('D')).dot(circulant(v('a')))
    from trees import *
    #t_d = tree_dist(Tree('D',[Tree('a',[])]))
    t_d = tree_dist(Tree.from_penn('(S (D a) (E b))'))
    #t_d = tree_dist(Tree.from_penn('(S (D a) (S (D a) (E b)))'))
    #print t_d[:,0].dot(Dw0[:,0])
    print Pd[:,0].dot(t_d[:,0])
    print Pd.dot(t_d)
    #Pd = Pd.dot(circulant(v('a')).T)
        # # print Pd
    Dp = test_P(parser,w)

    Dp = invsc(v('2')).dot(invsc(v('2'))).dot(Dp)

    print Dp[:,0].dot(t_d[:,0])
    #print circulant(v('a')).dot(circulant(v('a')).T)
        # #print Dp
    #print 'Pd: ',Pd[:,0].dot(sc(v('a'))[:,0])
    #print 'Dp: ',Dp[:,0].dot(sc(v('a'))[:,0])


    #print Pd[:,0].dot(Dp[:,0])

        #a = sc(v('0')).dot(sc(v('0'))).dot(sc(v('a'))).dot(sc(v('Sep')))
        #b = invsc(v('0')).dot(invsc(v('0'))).dot(a).dot(invsc(v('Sep')))
        #print a[:,0].dot(b[:,0])
        #print 'primo: ',b[:,0].dot(sc(v('a'))[:,0])
        # #print np.linalg.norm(Pd-Dp,2)
        # a = v('a')
        # b = v('0')
        # c = v('Sep')
        # #a2 = v('a')
        # #print np.linalg.norm(a-a2)
        # #print sc(a).dot(invsc(a)) - invsc(a).dot(sc(a))

        # #A = circulant(a)
        # #print A,'\n'
        # #print A.dot(A.T)
        # x1 = sc(b).dot(sc(a)).dot(sc(c))
        # x2 = invsc(b).dot(x1).dot(invsc(c))
        # print x2[:,0].dot(sc(a)[:,0])
