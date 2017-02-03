from parserNLP.Grammar import Grammar
from parserNLP.CYK import CYK
from vectors import *
#from keras_dt import *
from convolutions import *

dim = 1024
#dt = DT(dim=1024, lexicalized=True)
gen = Vector_generator(dim=dim)
Phi = permutation_matrices(dim)[1]

#print Phi

#[v]+
def sc(v):
    return circulant(v).dot(Phi)
#[v]-
def invsc(v):
    return sc(v).T

#initialization of level 0
def init(w):
    P = K.zeros((dim,)).eval()
    #print P[0]
    for i in range(len(w)):
        s = (sc(gen.get_random_vector('0')).dot(sc(gen.get_random_vector(str(i)))).dot(sc(gen.get_random_vector(w[i]))).dot(sc(gen.get_random_vector('Sep'))))
        P = P + s
    return P
#perterminal rules
def preterminals(P,G,w):
    R = np.array([0])
    #R=sum r_i preterminal
    # for i in range(len(D)):
    #     for chart in D[i,i]:
    for rule in G.get_unit_productions():
        #print 'prima: ',sc(gen.get_random_vector(rule.head())).dot(sc(gen.get_random_vector('Sep'))).dot(sc(gen.get_random_vector(rule.head())))
        #print 'circ: ',circulant(gen.get_random_vector(rule.production())),'rule: ',rule.production(),'vect: ',gen.get_random_vector(rule.production())
        R = R + (sc(gen.get_random_vector(rule.head())).dot(sc(gen.get_random_vector('Sep'))).dot(sc(gen.get_random_vector(rule.head()))).dot(circulant(gen.get_random_vector(rule.production()))).dot(invsc(gen.get_random_vector('Sep'))).dot(invsc(gen.get_random_vector(rule.head()))))
        #print R
    #print R
    for i in range(len(w)):
        s = (sc(gen.get_random_vector('1')).dot(sc(gen.get_random_vector(str(i)))).dot(R).dot(invsc(gen.get_random_vector(str(i)))).dot(invsc(gen.get_random_vector('0'))).dot(P))
        #print s
        P = P + s
    return P
#binary rules
def binary(P,G,w):
    for i in range(2,len(w)):
    	for j in range(0,len(w)-i+2):
    		Pa = K.zeros((dim,)).eval()

    		for rule in G.get_nonunit_productions():
    			RL = sc(gen.get_random_vector(rule[0])).dot(sc(gen.get_random_vector('Sep'))).dot(invsc(gen.get_random_vector(rule[1]))).dot(invsc(gen.get_random_vector(rule[0]))).dot(invsc(gen.get_random_vector(rule.head())))
    			RL_ = sc(gen.get_random_vector(rule.head())).dot(Phi).dot(invsc(gen.get_random_vector('Sep'))).dot(invsc(gen.get_random_vector(rule[0])))
    			RR = sc(gen.get_random_vector(rule.head())).dot(sc(gen.get_random_vector(rule[0]))).dot(sc(gen.get_random_vector(rule[1]))).dot(Phi).dot(invsc(gen.get_random_vector('Sep'))).dot(invsc(gen.get_random_vector(rule[1])))
    			RR_ = sc(gen.get_random_vector(rule[1])).dot(sc(gen.get_random_vector('Sep')))
    			#print RL,RL_,RR,RR_
    			for k in range(0,i+2):
    				#print k
    				Pa = Pa + RL_.dot(invsc(gen.get_random_vector(str(j)))).dot(invsc(gen.get_random_vector(str(k)))).dot(P).dot(RL).dot(RR).dot(invsc(gen.get_random_vector(str(j+k)))).dot(invsc(gen.get_random_vector(str(i-k)))).dot(P).dot(RR_)
    				#print Pa
    			P = P + sc(gen.get_random_vector(str(i))).dot(sc(gen.get_random_vector(str(j)))).dot(sc(gen.get_random_vector(rule.head()))).dot(sc(gen.get_random_vector('Sep'))).dot(Pa).dot(invsc(gen.get_random_vector('Sep'))).dot(invsc(gen.get_random_vector(rule.head())))
    return P

#transform P to P_dist with algo5,6
def cyk_dist(G,w):
	w = w.replace(' ','')
	P_dist = init(w)
	#print P_dist
	P_dist = preterminals(P_dist, G, w)
	#print P_dist
	#P_dist = binary(P_dist, G, w)
	return P_dist
'''
P : la matrice di CYK originale
Pd : la matrice di CYK distribuito
allora D(P) e' la versione distribuita di P
deve accadere che D(P) e Pd sono simili
come prima cosa
poi che una cella in P
sia reperibile in Pd
'''

#trasformazione di P in distributed with trees
'''ogni cella ha gli alberi che rappresenta
e dunque puoi generare il contenuto della cella in maniera distribuita
facendo le stesse operazioni
per codificare l'albero
e unendoci i vettori dei due indici'''
def tree_dist(t):
    if len(t) == 0:
        return sc(gen.get_random_vector(t.label))
    s = sc(gen.get_random_vector(t.label))
    for child in t:
        s = s.dot(tree_dist(child))
    return s

def test_P(parser,w):
    w = w.replace(' ','')
    Dp = np.array([0])
    #test

    for i in range(len(w)):
        s = sc(gen.get_random_vector('0')).dot(sc(gen.get_random_vector(str(i)))).dot(sc(gen.get_random_vector(w[i])))
        Dp = Dp + s

    #first row
    for i in range(len(parser.C)):
        #preterminal trees
        #print chart
        for A in parser.C[i,i]:
            tree = parser.get_tree(A)
            #print 'tree: ',tree
            td = sc(gen.get_random_vector("1")).dot(sc(gen.get_random_vector(str(i)))).dot(tree_dist(tree))
            Dp = Dp + td
    #generic row
    '''for i in range(2,len(w)):
    	for j in range(0,len(w)-i+2):
    		if i==j:
    			continue
    		for A in parser.C[j,i]:
    			#print A
    			tree = parser.get_tree(A)
                #print 'tree: ',tree
                td = sc(gen.get_random_vector(str(i))).dot(sc(gen.get_random_vector(str(j)))).dot(tree_dist(tree))
                #print td
                Dp = Dp + td'''
    return Dp

'''
con grammatiche stupide e frasi piccole
e con grammatiche piu' complesse e frasi piu' lunghe
'''
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
        #     '''a = sc(gen.get_random_vector('0')).dot(sc(gen.get_random_vector('0'))).dot(sc(gen.get_random_vector('a'))).dot(sc(gen.get_random_vector('Sep')))
        #     a_0 = invsc(gen.get_random_vector('0')).dot(a)
        #     a_1 = sc(gen.get_random_vector('0')).dot(sc(gen.get_random_vector('a'))).dot(sc(gen.get_random_vector('Sep')))
        #     print a_0
        #     print a_1
        #     print np.linalg.norm(a_0-a_1,2)'''
	    
    	   #  a = gen.get_random_vector('a')
    	   #  A = circulant(a)
    	   #  print A.dot(A.T)
            #Pd = cyk_dist(G,w)
            #print sc(gen.get_random_vector("0")),sc(gen.get_random_vector("0"))
            #print np.linalg.norm(sc(gen.get_random_vector("Sep"))-invsc(gen.get_random_vector("Sep")),2)
            #Pd = invsc(gen.get_random_vector('0')).dot(invsc(gen.get_random_vector('0'))).dot(Pd)
            #Pd = Pd.dot(invsc(gen.get_random_vector('Sep')))
            #print Pd
            #Dp = test_P(parser,w)
            #Dp = invsc(gen.get_random_vector('0')).dot(invsc(gen.get_random_vector('0'))).dot(Dp)
            #print Dp
            #print np.linalg.norm(Pd-Dp,2)
            #print invsc(gen.get_random_vector("0")).dot(invsc(gen.get_random_vector("1"))).dot(Pd)
            #print invsc(gen.get_random_vector("0")).dot(invsc(gen.get_random_vector("1"))).dot(Dp)
    #else:
    Pd = cyk_dist(G,w)
    Pd = invsc(gen.get_random_vector("0")).dot(invsc(gen.get_random_vector("1"))).dot(Pd)
    Pd = Pd.dot(invsc(gen.get_random_vector('Sep')))
        # # print Pd
    Dp = test_P(parser,w)
    Dp = invsc(gen.get_random_vector("0")).dot(invsc(gen.get_random_vector("1"))).dot(Dp)
        # #print Dp
    print 'Pd: ',Pd[:,0].dot(sc(gen.get_random_vector('a')).dot(sc(gen.get_random_vector('D')))[:,0])
    print 'Dp: ',Dp[:,0].dot(sc(gen.get_random_vector('a')).dot(sc(gen.get_random_vector('D')))[:,0])


    print Pd[:,0].dot(Dp[:,0])

        #a = sc(gen.get_random_vector('0')).dot(sc(gen.get_random_vector('0'))).dot(sc(gen.get_random_vector('a'))).dot(sc(gen.get_random_vector('Sep')))
        #b = invsc(gen.get_random_vector('0')).dot(invsc(gen.get_random_vector('0'))).dot(a).dot(invsc(gen.get_random_vector('Sep')))
        #print a[:,0].dot(b[:,0])
        #print 'primo: ',b[:,0].dot(sc(gen.get_random_vector('a'))[:,0])
        # #print np.linalg.norm(Pd-Dp,2)
        # a = gen.get_random_vector('a')
        # b = gen.get_random_vector('0')
        # c = gen.get_random_vector('Sep')
        # #a2 = gen.get_random_vector('a')
        # #print np.linalg.norm(a-a2)
        # #print sc(a).dot(invsc(a)) - invsc(a).dot(sc(a))

        # #A = circulant(a)
        # #print A,'\n'
        # #print A.dot(A.T)
        # x1 = sc(b).dot(sc(a)).dot(sc(c))
        # x2 = invsc(b).dot(x1).dot(invsc(c))
        # print x2[:,0].dot(sc(a)[:,0])