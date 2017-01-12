from parserNLP.Grammar import Grammar
from parserNLP.CYK import CYK
from vectors import *
from keras_dt import *
from convolutions import *
dim = 1024
gen = Vector_generator(dim=dim)
#[v]+
def sc(v):
    if type(v) != np.ndarray:
        v=v.eval()
    Phi = permutation_matrices(dim)[1]
    #print Phi.shape
    return circulant(v).dot(Phi)
#[v]-
def invsc(v):
    return sc(v).T
#initialization of level 0
def init(w):
    P = K.zeros((dim,)).eval()
    #print P[0]
    for i in range(len(w)):
        s = (sc(gen.get_random_vector('0')).dot(sc(gen.get_random_vector(str(i)))).dot(sc(gen.get_random_vector(w[i]))).dot(sc(gen.get_random_vector('Sep'))).dot(np.eye(1,dim,0)[0]))
        #print s
        P = P + s
    return P
#perterminal rules
def preterminals(P,D,w):
    R = np.array([0])
    #R=sum r_i preterminal
    for i in range(len(D)):
        for chart in D[i,i]:
            R = R + (sc(gen.get_random_vector(chart.rule.head())).dot(circulant(gen.get_random_vector(chart.rule.production()))).dot(invsc(gen.get_random_vector('Sep'))).dot(invsc(gen.get_random_vector(chart.rule.production()))))
    
    for i in range(len(w)):
        s = (sc(gen.get_random_vector('1')).dot(sc(gen.get_random_vector(str(i)))).dot(R).dot(invsc(gen.get_random_vector(str(i)))).dot(invsc(gen.get_random_vector('0'))).dot(P))
        P = P + s
    return P
#binary rules
#TODO
def binary(P,D,w):
    for i in range(2,len(w)):
    	for j in range(0,len(w)-i+1):
    		Pa=np.array([0])
    		for A in D[j,i]:
    			RL = sc(gen.get_random_vector(A.rule.production()[0])).dot(sc(gen.get_random_vector('Sep'))).dot(invsc(gen.get_random_vector(A.rule.production()[1]))).dot(invsc(gen.get_random_vector(A.rule.production()[0]))).dot(invsc(gen.get_random_vector(invsc(A.rule.head()))))
    			RL_ = 
    			RR =
    			RR_ =
    			for k in range(0,i+1):
    				Pa = Pa +
    return P
def cyk_dist(D,w):
	w = w.replace(' ','')
	P_dist = init(w)
	#print P_dist
	P_dist = preterminals(P_dist, D, w)
	binary(P_dist, D, w)
	return P_dist
w = 'a a b'
G = Grammar('S')
G.add_rules_from_file('gramm_l')
parser = CYK(G)
parser.parse(w)
P = parser.C
print P
if K.backend() == 'tensorflow':
	sess = K.tf.Session()
	K.set_session(sess)
	with sess.as_default():
		print cyk_dist(P,w)
else:
	print cyk_dist(P,w)