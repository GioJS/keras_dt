from parserNLP.Grammar import Grammar
from parserNLP.CYK import CYK
from vectors import *
from convolutions import *
import json

conf = None
with open('conf.json') as f:
    conf = json.load(f)

if conf is not None:
    dim = conf['dim']
    sig_param = conf['sig']
    displacement = conf['disp']
else:
    dim = 1024
    sig_param = 36
    displacement = 0.5
gen = Vector_generator(dim=dim)
Phi = permutation_matrices(dim)[1]
# print("Permutatation: ", Phi[0:10])
v = gen.get_random_vector


# [v]+
def sc(v):
    return circulant(v).dot(Phi)


#    return circulant(v) #v2

# v2
def v2disp(i):
    return i + 0


# [v]-
def invsc(v):
    return sc(v).T


# init

index0 = sc(v('0'))
index1 = sc(v('1'))


# index2 = sc(v('2'))
# index3 = sc(v('3'))
# index4 = sc(v('4'))

# D = sc(v('D'))
# E = sc(v('E'))
# S = sc(v('S'))
# sep = sc(v('Sep'))


def sigmoid(x):
    return 1 / (1 + np.exp(-(x - displacement) * sig_param))


def init_simple(w):
    P = np.array([0])
    for i in range(len(w)):
        s = index0.dot(sc(v(str(v2disp(i + 1))))).dot(sc(v(w[i])))
        P = P + s
    return P


# perterminal rules
def preterminals_simple(P, G, w):
    R = np.array([0])

    for rule in G.get_unit_productions():
        # print rule
        R = R + sc(v(rule.head())).dot(invsc(v(rule.production())))

    s = np.array([0])
    for i in range(len(w)):
        s = s + index1.dot(sc(v(str(i + 1)))).dot(R).dot(invsc(v(str(i + 1)))).dot(index0.T).dot(P)
    P = s
    return P


def preterminals_simple_with_sigmoid(P, G, w, symbols):  # v2
    if symbols == {}:
        for rule in G.get_unit_productions():
            symbols[rule.head()] = np.zeros((dim, dim))

        for rule in G.get_unit_productions():
            symbols[rule.head()] = symbols[rule.head()] + invsc(v(rule.production()))

    s = np.zeros((dim, dim))
    for i in range(len(w)):
        # print("Symbol = " , i, w[i])
        tmp = np.zeros((dim, dim))
        for symbol in G.groups:
            if symbol in symbols:
                # print("QUESTO >>> ", symbol)
                detect_matrix = sigmoid(symbols[symbol].dot(invsc(v(str(v2disp(i + 1))))).dot(index0.T).dot(P))
                detect_matrix = np.multiply(detect_matrix, np.identity(dim))
                # print(detect_matrix[0:4,0:4])
                tmp = tmp + sc(v(symbol)).dot(detect_matrix)

        s = s + index1.dot(sc(v(str(v2disp(i + 1))))).dot(tmp)
    return s


def compute_R_simple(G, rules_A):
    Ra = np.zeros((dim, dim))
    for i in rules_A:
        rule = G[i]
        if not rule.is_preterminal():
            # print rule
            Ra = Ra + invsc(v(rule[0])).dot(invsc(v(rule[1])))
    return Ra


def binary_simple(P, G, w):
    n = len(w)
    R = {}
    keys = sorted(G.groups)
    # keys = G.groups
    # G.groups = non-terminals
    for A in keys:
        rules_A = G.get_rules(A)
        R[A] = compute_R_simple(G, rules_A)

    for i in range(2, n + 1):
        for j in range(1, n - i + 2):
            for A in keys:
                Ra = R[A]
                Pa = np.zeros((dim, dim))

                for k in range(1, i):
                    pre = invsc(v(str(v2disp(j)))).dot(invsc(v(str(k))).dot(P)).dot(Ra).dot(
                        invsc(v(str(v2disp(j + k)))).dot(invsc(v(str(i - k)))).dot(P))

                    sig = sigmoid(pre)
                    # trick
                    eye = np.multiply(sig, np.identity(dim))

                    Pa = Pa + eye

                s = sc(v(str(i))).dot(sc(v(str(v2disp(j))))).dot(sc(v(A))).dot(Pa)

                P = P + s

    return P


# transform P to P_dist with algo5,6
def cyk_dist_simple(G, w, symbols):
    w = w.split(' ')  # now terminals are strings
    P_dist = init_simple(w)
    # print P_dist
    P_dist = preterminals_simple_with_sigmoid(P_dist, G, w, symbols)
    # print P_dist
    P_dist = binary_simple(P_dist, G, w)
    return P_dist.astype(dtype=np.float32)


# initialization of level 0
def init(w):
    P = np.array([0])
    for i in range(len(w)):
        s = index0.dot(sc(v(str(i + 1)))).dot(sc(v(w[i]))).dot(sep)
        P = P + s
    return P


# perterminal rules
def preterminals(P, G, w):
    R = np.array([0])

    for rule in G.get_unit_productions():
        # print rule
        R = R + (
            sc(v(rule.head())).dot(sep).dot(sc(v(rule.head()))).dot(sep).dot(sc(v(rule.production()))).dot(sep.T).dot(
                invsc(v(rule.head())))).dot(sep.T).dot(invsc(v(rule.production())))

    s = np.array([0])
    for i in range(len(w)):
        s = s + index1.dot(sc(v(str(i + 1)))).dot(R).dot(invsc(v(str(i + 1)))).dot(index0.T).dot(P)
    P = s
    return P


def compute_C(G):
    C = np.array([0])
    for A in G.groups:
        C = C + sc(v(A)).dot(sep)
    return C


def compute_R(G, rules_A):
    Ra = np.zeros((dim, dim))
    for i in rules_A:
        rule = G[i]
        if not rule.is_preterminal():
            # print rule
            Ra = Ra + sc(v(rule[0])).dot(sep).dot(Phi).dot(sep.T).dot(invsc(v(rule[1])))
    return Ra


def binary(P, G, w):
    n = len(w)
    C = compute_C(G)
    # s = np.array([0])
    R = {}
    print('preterminal: \n', P)
    # G.groups = non-terminals
    for A in G.groups:
        rules_A = G.get_rules(A)
        R[A] = compute_R(G, rules_A)
    for i in range(2, n + 1):
        for j in range(1, n - i + 2):
            # print i,j
            for A in G.groups:
                Ra = R[A]
                # print 'A: ',A,'Ra: \n',Ra
                Pa = np.array([0])
                for k in range(1, i):
                    # print "j,k:",j,k
                    # print "j+k,i-k",j+k,i-k
                    Pa = Pa + C.dot((invsc(v(str(j))))).dot(invsc(v(str(k)))).dot(P).dot(Ra).dot(
                        invsc(v(str(j + k)))).dot(invsc(v(str(i - k)))).dot(P).dot(C.T)
                # print 'Pa:\n',Pa
                # print (Pa==0).all()
                s = sc(v(str(i))).dot(sc(v(str(j)))).dot(sc(v(A))).dot(sep).dot(sc(v(A))).dot(sep).dot(Pa).dot(
                    sep.T).dot(invsc(v(A)))
                # print 's: \n',s
                P = P + s
                # print 'P new: \n',P
    # P = P + s
    return P


# transform P to P_dist with algo5,6
def cyk_dist(G, w):
    w = w.replace(' ', '')
    P_dist = init(w)
    # print P_dist
    P_dist = preterminals(P_dist, G, w)
    # print P_dist
    P_dist = binary(P_dist, G, w)
    return P_dist


# trasformazione di P in distributed with trees
def tree_dist(t):
    if len(t) == 0:
        return sc(v(t.label))
    s = sc(v(t.label)).dot(sep)
    for child in t:
        s = s.dot(tree_dist(child))
    return s


def test_P(parser, w):
    w = w.replace(' ', '')
    Dp = np.array([0])
    # test

    for i in range(len(w)):
        s = index0.dot(sc(v(str(i + 1)))).dot(sc(v(w[i])))
        Dp = Dp + s

    # first row
    for i in range(len(parser.C)):
        # preterminal trees
        # print chart
        for A in parser.C[i, i]:
            tree = parser.get_tree(A)
            # print 'tree: ',tree
            td = index1.dot(sc(v(str(i + 1)))).dot(tree_dist(tree))
            Dp = Dp + td
    # generic row
    for i in range(2, len(w)):
        for j in range(0, len(w) - i + 2):
            if i == j:
                continue
            for A in parser.C[j, i]:
                # print A
                # print i,j
                tree = parser.get_tree(A)
                # print 'tree: ',tree
                # P is 3x3 Dp is 4x3
                # print i+1-j, j+1, A
                td = sc(v(str(i + 1 - j))).dot(sc(v(str(j + 1)))).dot(tree_dist(tree))
                # print td
                Dp = Dp + td
    return Dp


if __name__ == '__main__':
    G = Grammar('S')
    G.add_rules_from_file('gramm_l')
    '''print(G.get_unit_productions())
    print()
    print(G.get_nonunit_productions())'''
    # parser = CYK(G)
    #    for i in range(2,3):
    # w = ('a '*i)+'b'
    w = 'a a a b'
    # w = 'john kiss a girl'
    print(w)
    # parser.parse(w)
    # print(parser.C)
    # print(parser.getTrees())
    # exit(0)
    # parser.parse(w)
    # P = parser.C
    # print(P)
    # exit(0)
    from trees import *

    dist_P = cyk_dist_simple(G, w)
    # print(dist_P)
    '''print('NP\n',invsc(v('NP')).dot(index1.T).dot(index1.T).dot(dist_P))
    print('V\n',invsc(v('V')).dot(index2.T).dot(index1.T).dot(dist_P))
    print('Det\n',invsc(v('Det')).dot(index3.T).dot(index1.T).dot(dist_P))
    print('N\n',invsc(v('N')).dot(index4.T).dot(index1.T).dot(dist_P))
    print('NP\n',invsc(v('NP')).dot(index3.T).dot(index2.T).dot(dist_P))
    print('VP\n',invsc(v('VP')).dot(index2.T).dot(index3.T).dot(dist_P))
    print('S\n',invsc(v('S')).dot(index1.T).dot(index4.T).dot(dist_P))'''

    print('D1\n', D.T.dot(index1.T).dot(index1.T).dot(dist_P))
    print('D2\n', D.T.dot(index2.T).dot(index1.T).dot(dist_P))
    print('D3\n', D.T.dot(index3.T).dot(index1.T).dot(dist_P))
    print('E\n', E.T.dot(index4.T).dot(index1.T).dot(dist_P))
    print('S1\n', S.T.dot(index3.T).dot(index2.T).dot(dist_P))
    print('S2\n', S.T.dot(index2.T).dot(index3.T).dot(dist_P))
    print('S3\n', S.T.dot(index1.T).dot(index4.T).dot(dist_P))

    '''pure_P = index1.dot(index1).dot(D)
    pure_P += index1.dot(index2).dot(D)
    pure_P += index1.dot(index3).dot(E)
    #print(pure_P)
    pre = init_simple(w)
    pre_1 = preterminals_simple_with_sigmoid(pre, G, w)
    pre_2 = preterminals_simple(pre, G, w)
    #print(pre)

    #print(S.T.dot(index2.T).dot(index2.T).dot(dist_P))
    #print(index1.T.dot(index1.T).dot(dist_P).dot(D.T))
    R =  D.T.dot(S.T)
    print sigmoid(index2.T.dot(index1.T).dot(pure_P).dot(R).dot(index3.T).dot(index1.T).dot(pure_P))

    R =  D.T.dot(E.T)
    print("\n\nPURE\n", sigmoid(index2.T.dot(index1.T).dot(pure_P).dot(R).dot(index3.T).dot(index1.T).dot(pure_P))[0:4,0:4])
    print("\n\nCOMPUTED SIGMOID\n", sigmoid(index2.T.dot(index1.T).dot(pre_1).dot(R).dot(index3.T).dot(index1.T).dot(pre_1))[0:4,0:4] )
    print("\n\nCOMPUTED NO SIGMOID\n", sigmoid(index2.T.dot(index1.T).dot(pre_2).dot(R).dot(index3.T).dot(index1.T).dot(pre_2))[0:4,0:4] )'''
