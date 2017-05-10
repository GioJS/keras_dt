from cyk_dist import *
import os


# parsing with classical cyk
def getP(w, G):
    parser = CYK(G)
    parser.parse(w)
    return parser.C


# parsing with distributed cyk
def getPDistributed(w, G, symbols, R):
    return cyk_dist_simple(G, w, symbols, R)


# get active rules
# def getRules(w, P):
#     active_rules = []
#     print(len(w))
#     # print('l',len(P))
#     # print('ll',len(P[0]))
#     for i in range(len(w)):
#         rule = P[i, i]
#         if len(rule) > 0:
#             active_rules.append((1, i + 1, rule))
#
#     for i in range(2, len(w)):
#         # print('i',i)
#         for j in range(0, len(w) - i + 2):
#             # print('j',j)
#
#             rule = P[j, i]
#
#             if len(rule) > 0 and not (i + 1 - j, j + 1, rule) in active_rules:
#                 active_rules.append((i + 1 - j, j + 1, rule))
#     return active_rules


def tranform_P(P, w):
    new_P = []
    row = []
    for i in range(len(w)):
        row.append(w[i])
    new_P.append(row)
    for i in range(len(w)):
        empty_row = []
        for j in range(len(w)):
            empty_row.append([])
        new_P.append(list(empty_row))

    for i in range(len(P)):
        for A in P[i, i]:
            tree = A.rule.head()
            new_P[1][i].append(tree)
            # active_rules.append((1, i + 1, rule))
    # generic row
    for i in range(0, len(w)):
        for j in range(0, len(w)):
            # print(i,j)
            if i != j:
                for A in P[j, i]:
                    tree = A.rule.head()
                    # td = sc(v(str(i + 1 - j))).dot(sc(v(str(j + 1)))).dot(sc(v(tree)))
                    # print(i + 1 - j, j + 1)
                    new_P[i - j + 1][j].append(tree)
    return new_P


def test_P(P, w):
    Dp = np.array([0])
    # test

    # for i in range(len(w)):
    #    s = index0.dot(sc(v(str(i + 1)))).dot(sc(v(w[i])))
    #    Dp = Dp + s

    # first row
    for i in range(len(P)):
        for A in P[i, i]:
            tree = A.rule.head()
            # print 'tree: ',tree
            td = index1.dot(sc(v(str(v2disp(i + 1))))).dot(sc(v(tree)))
            Dp = Dp + td
    # generic row
    for i in range(0, len(w)):
        for j in range(0, len(w)):
            if i != j:
                for A in P[j, i]:
                    tree = A.rule.head()
                    # print ('t', tree)
                    # print('ij', i + 1 - j, j + 1)
                    td = sc(v(str(i + 1 - j))).dot(sc(v(str(v2disp(j + 1))))).dot(sc(v(tree)))
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
files = {'l': 'gramm_l', 'm': 'gramm_m', 'ml': 'gramm_ml', 'm1': 'gramm_m10', 'm2': 'gramm_m100', 'mpp':
    'gramm_pp_m10'}
print(conf)
if conf is not None:
    sentences = conf['sentences']
    file = files[conf['grammar']]
    med = conf['med']
    name_exp = conf['name']
else:
    file = files['m']
    w = 'john likes a girl'

# print(w)
symbols = {}
G = Grammar('S')
G.add_rules_from_file(file)
parser = CYK(G)
p_means = []
r_means = []
R = {}
'''import random

random.shuffle(sentences)'''
s = 0

for w in sentences:
    precisions = [0] * dim
    recalls = [0] * dim
    intersections = [0] * dim
    system_decisions = [0] * dim
    oracle_request = [0] * dim
    P = getP(w, G)
    print("This is the matrix P\n", P)
    # new_P = tranform_P(P, w.split())
    # print("This is the matrix P_new\n", new_P)
    P_dist = getPDistributed(w, G, symbols, R)
    # print(symbols)
    P_real = test_P(P, w.split())
    n = G.get_unit_productions()

    for j in range(dim):
        # precisions.append(P_dist[:, i].dot(P_real[:, i]) / P_dist[:, i].dot(P_dist[:, i]))
        intersections[j] = intersections[j] + P_dist[:, j].dot(P_real[:, j])
        system_decisions[j] = system_decisions[j] + P_dist[:, j].dot(P_dist[:, j])
        oracle_request[j] = oracle_request[j] + P_real[:, j].dot(P_real[:, j])
        # recalls.append(P_dist[:, i].dot(P_real[:, i]) / P_real[:, i].dot(P_real[:, i]))
    ####
    for i in range(dim):
        precisions[i] = intersections[i] / system_decisions[i]
        recalls[i] = intersections[i] / oracle_request[i]
    print(precisions)
    print(recalls)
    if not os.path.exists('./experiments'):
        os.mkdir('experiments')
    np.savetxt("experiments/precisions_%s_%d.txt" % (name_exp, s), precisions)
    np.savetxt("experiments/recalls_%s_%d.txt" % (name_exp, s), recalls)
    s += 1
    # precisions.sort()
    # recalls.sort()
    # precisions = precisions[med:-med]
    # recalls = recalls[med:-med]
    # if np.mean(precisions) > 2:
    #     print('sentence: ', w)
    #     p = np.array(precisions)
    #     errors = p[p >= 2]


    p_means.append(np.mean(precisions))
    # r_means.append(np.mean(recalls))
    # print(p_means, r_means)
print('Precision: ', np.mean(p_means))
print('Recall: ', np.mean(r_means))
