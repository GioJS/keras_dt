from parserNLP.Grammar import Grammar
from parserNLP.CYK import CYK
from vectors import *
from convolutions import *
import json
import warnings
from scipy.special import expit

warnings.filterwarnings('error')

''' This is the implementation of the novel version of the cyk parser with the input of Giorgio Satta  '''




class DistributedCYKParser:
    def __init__(self):
        conf = None
        with open('conf.json') as f:
            conf = json.load(f)

        if conf is not None:
            self._dim = conf['dim']
            self._sig_param = conf['sig']
            self._displacement = conf['disp']
        else:
            self._dim = 1024
            self._sig_param = 36
            self._displacement = 0.5
        self._gen = Vector_generator(dim=self._dim)
        self._Phi = permutation_matrices(self._dim)[1]
        # print("Permutatation: ", Phi[0:10])
        self._v = self._gen.get_random_vector

        self._index0 = self.sc(self._v('0'))
        self._index1 = self.sc(self._v('1'))


    def sigmoid(self,x):
        try:
            y = expit((x - self._displacement) * self._sig_param)
        except Warning:
            print(np.max(x))
            print(sys.exc_info()[0])
        return y

    def SC(self,symbol):
        if type(symbol) is not str:
            symbol = str(symbol)
        return self.sc(self._v(symbol))

    def INVSC(self,symbol):
        if type(symbol) is not str:
            symbol = str(symbol)
        return self.invsc(self._v(symbol))


    # [v]+
    def sc(self,v):
        return circulant(v).dot(self._Phi)

    # [v]-
    def invsc(self,v):
        return self.sc(v).T

    def dist_cell(self, i , j):
        return self.sc(self._v(str(i))).dot(self.sc(self._v(str(j))))

    def inv_dist_cell(self,i,j):
        return self.invsc(self._v(str(i))).dot(self.invsc(self._v(str(j))))

    def generate_distributed_preterminal_rules(self,G):
        self._symbols= {}
        for rule in G.get_unit_productions():
            self._symbols[rule.head()] = np.zeros(self._dim)
        for rule in G.get_unit_productions():
            self._symbols[rule.head()] = self._symbols[rule.head()] + self._v(rule.production())
        for s in self._symbols:
            self._symbols[s] = self.invsc(self._symbols[s])

    def generate_distributed_binary_rules(self,G):
        keys = sorted(G.groups)
        self._R = {}
        # keys = G.groups
        # G.groups = non-terminals
        for A in keys:
            rules_A = G.get_rules(A)
            self._R[A] = np.zeros((self._dim, self._dim))
            for i in rules_A:
                rule = G[i]
                if not rule.is_preterminal():
                    # print rule
                    symb1 = rule[0]
                    symb2 = rule[1]
                    self._R[A] = self._R[A] + self.INVSC(rule[0]).dot(self.INVSC(rule[1]))

    def initialize_distributed_grammar(self,G):
        self.generate_distributed_preterminal_rules(G)
        self.generate_distributed_binary_rules(G)
        self._grammar_symbols = sorted(G.groups)


    def initialize_words_in_matrix(self, w):
        P = np.array([0])
        for i in range(len(w)):
            s = self.SC(i).dot(self.SC(i+1)).dot(self.SC(w[i]))
            P = P + s
        return P

    def preterminals_simple_with_sigmoid(self, P, w):  # v2

        P_left = np.zeros((self._dim, self._dim))
        P_right = np.zeros((self._dim, self._dim))
        for i in range(len(w)):
            # print("Symbol = " , i, w[i])
            tmp = np.zeros((self._dim, self._dim))
            for symbol in self._symbols:
                # print("QUESTO >>> ", symbol)
                detect_matrix = self.sigmoid(self._symbols[symbol].dot(self.INVSC(i+1)).dot(self.INVSC(i)).dot(P))
                detect_matrix = np.multiply(detect_matrix, np.identity(self._dim))
                # print(detect_matrix[0:4,0:4])
                tmp = tmp + self.SC(symbol).dot(detect_matrix)
            P_left = P_left + self.INVSC(i).dot(self.INVSC(i+1)).dot(tmp)
            P_right = P_right + tmp.dot(self.SC(i)).dot(self.SC(i+1))
        return (P_left,P_right)

    def binary_simple(self, P, w):
        n = len(w)
        (P_left,P_right) = P
        for j in range(2, n + 1):
            for i in range(j-2, -1,-1):
                for A in self._grammar_symbols:
                    Ra = self._R[A]
                    Pa = np.multiply(self.sigmoid(self.INVSC(j).dot(self.SC(i)).dot(P_left).dot(Ra).dot(P_right)), np.identity(self._dim))
                    P_left = P_left + self.INVSC(i).dot(self.INVSC(j)).dot(self.SC(A)).dot(Pa)
                    P_right = P_right + self.SC(A).dot(self.SC(i)).dot(self.SC(j)).dot(Pa)
        return (P_left,P_right)

    def checkContent(self,P,w):
        (P_left, P_right) = P
        for i in range(0,len(w)+1):
            for j in range(0,len(w)+1):
                for symbol in self._grammar_symbols:
                    a = self.invsc(self._v(symbol)).dot(self.dist_cell(j,i)).dot(P_left)
                    b = self.inv_dist_cell(j,i).dot(self.invsc(self._v(symbol))).dot(P_right)
                    print("Check")

    def parse(self,w):
        w = w.split(' ')
        P = self.initialize_words_in_matrix(w)
        P = self.preterminals_simple_with_sigmoid(P,w)
        P = self.binary_simple(P,w)
        return P


if __name__ == '__main__':


    G = Grammar('S')
    G.add_rules_from_file('gramm_l')

    dp = DistributedCYKParser()

    dp.initialize_distributed_grammar(G)

    prova = dp.dist_cell(1,2).dot(dp.inv_dist_cell(2,1))

    w = 'a a b'
    # w = 'john kiss a girl'
    print(w)

    (P_left,P_right) =  dp.parse(w)

    print(dp.sc(dp._v("D")).dot(dp.dist_cell(2,1)).dot(P_left))

    # parser.parse(w)
    # print(parser.C)
    # print(parser.getTrees())
    # exit(0)
    # parser.parse(w)
    # P = parser.C
    # print(P)
    # exit(0)
    from trees import *

    # print(dist_P)
    '''print('NP\n',invsc(v('NP')).dot(index1.T).dot(index1.T).dot(dist_P))
    print('V\n',invsc(v('V')).dot(index2.T).dot(index1.T).dot(dist_P))
    print('Det\n',invsc(v('Det')).dot(index3.T).dot(index1.T).dot(dist_P))
    print('N\n',invsc(v('N')).dot(index4.T).dot(index1.T).dot(dist_P))
    print('NP\n',invsc(v('NP')).dot(index3.T).dot(index2.T).dot(dist_P))
    print('VP\n',invsc(v('VP')).dot(index2.T).dot(index3.T).dot(dist_P))
    print('S\n',invsc(v('S')).dot(index1.T).dot(index4.T).dot(dist_P))'''

    '''print('D1\n', D.T.dot(index1.T).dot(index1.T).dot(dist_P))
    print('D2\n', D.T.dot(index2.T).dot(index1.T).dot(dist_P))
    print('D3\n', D.T.dot(index3.T).dot(index1.T).dot(dist_P))
    print('E\n', E.T.dot(index4.T).dot(index1.T).dot(dist_P))
    print('S1\n', S.T.dot(index3.T).dot(index2.T).dot(dist_P))
    print('S2\n', S.T.dot(index2.T).dot(index3.T).dot(dist_P))
    print('S3\n', S.T.dot(index1.T).dot(index4.T).dot(dist_P))'''

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
