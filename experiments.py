from cyk_dist import *

#parsing with classical cyk
def getP(w,G):
    parser = CYK(G)
    parser.parse(w)
    return parser.C

#parsing with distributed cyk
def getPDistributed(w,G):
    return cyk_dist_simple(G,w)
#get active rules
def getRules(w,P):
    active_rules = []
    print(len(w))
    #print('l',len(P))
    #print('ll',len(P[0]))
    for i in range(len(w)):
        rule = P[i,i]
        if len(rule)>0:
            active_rules.append((1, i + 1, rule))

    for i in range(2,len(w)):
        #print('i',i)
        for j in range(0,len(w)-i+2):
            #print('j',j)

            rule = P[j, i]

            if len(rule)>0:
                active_rules.append((i+1-j, j+1, rule))
    return active_rules
#check if active rules are in P_dist
def checkInDist(active_rules, P_dist):
    symbols = len(active_rules)
    dist_symbols = 0
    for rules in active_rules:
        i = str(rules[0])
        j = str(rules[1])
        rule_l = rules[2]
        #print(rule_l,type(rule_l))
        head = rule_l[0].rule.head()
        print(i,j,head)
        selected = invsc(v(head)).dot(invsc(v(j))).dot(invsc(v(i))).dot(P_dist)
        sim = np.sum(np.diag(selected))/dim
        print(sim)
        if 0.8 <= sim <= 1.0:
            dist_symbols += 1
    return dist_symbols/symbols if symbols > 0 else 0.0


file = 'gramm_m'
w = 'jonh likes a girl'
G = Grammar('S')
G.add_rules_from_file(file)

P = getP(w, G)
print(P)
P_dist = getPDistributed(w, G)

active_rules = getRules(w.split(' '),P)

print(active_rules)
print(checkInDist(active_rules, P_dist))