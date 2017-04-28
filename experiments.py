from cyk_dist import *


# parsing with classical cyk
def getP(w, G):
    parser = CYK(G)
    parser.parse(w)
    return parser.C


# parsing with distributed cyk
def getPDistributed(w, G):
    return cyk_dist_simple(G, w)


# get active rules
def getRules(w, P):
    active_rules = []
    print(len(w))
    # print('l',len(P))
    # print('ll',len(P[0]))
    for i in range(len(w)):
        rule = P[i, i]
        if len(rule) > 0:
            active_rules.append((1, i + 1, rule))

    for i in range(2, len(w)):
        # print('i',i)
        for j in range(0, len(w) - i + 2):
            # print('j',j)

            rule = P[j, i]

            if len(rule) > 0 and not (i + 1 - j, j + 1, rule) in active_rules:
                active_rules.append((i + 1 - j, j + 1, rule))
    return active_rules


def test_P(P, w):
    Dp = np.array([0])
    # test

    for i in range(len(w)):
        s = index0.dot(sc(v(str(i + 1)))).dot(sc(v(w[i])))
        Dp = Dp + s

    # first row
    for i in range(len(P)):
        for A in P[i, i]:
            tree = A.rule.head()
            # print 'tree: ',tree
            td = index1.dot(sc(v(str(i + 1)))).dot(sc(v(tree)))
            Dp = Dp + td
    # generic row
    for i in range(2, len(w)):
        for j in range(0, len(w) - i + 2):
            if i == j:
                continue
            for A in P[j, i]:
                tree = A.rule.head()

                td = sc(v(str(i + 1 - j))).dot(sc(v(str(j + 1)))).dot(sc(v(tree)))
                Dp = Dp + td
    return Dp


# check if active rules are in P_dist
# def checkInDist(active_rules, P_dist):
#     symbols = len(active_rules)
#     dist_symbols = 0
#     eps = 9e-02
#     for rules in active_rules:
#         i = str(rules[0])
#         j = str(rules[1])
#         rule_l = rules[2]
#         # print(rule_l,type(rule_l))
#         head = rule_l[0].rule.head()
#         print(i, j, head)
#         selected = invsc(v(head)).dot(invsc(v(j))).dot(invsc(v(i))).dot(P_dist)
#         print(selected)
#         diag = np.diag(selected)
#         # print(sim)
#         if np.all(diag >= 0.5) and np.all(diag <= (1 + eps)):
#             dist_symbols += 1
#     return dist_symbols / symbols if symbols > 0 else 0.0


# l dummy grammar, m simple english grammar, ml a more complex english grammar
files = {'l': 'gramm_l', 'm': 'gramm_m', 'ml': 'gramm_ml'}
print(conf)
if conf is not None:
    w = conf['sentences'][0]
    file = files[conf['grammar']]
else:
    file = files['m']
    w = 'john likes a girl'
# print(w)
G = Grammar('S')
G.add_rules_from_file(file)
parser = CYK(G)
P = getP(w, G)
print(P)
P_dist = getPDistributed(w, G)
P_real = test_P(P, w.split())
precision = P_dist[:, 0].dot(P_real[:, 0]) / P_dist[:, 0].dot(P_dist[:, 0])
recall = P_dist[:, 0].dot(P_real[:, 0]) / P_real[:, 0].dot(P_real[:, 0])
print('Precision: ', precision)
print('Recall: ', recall)
