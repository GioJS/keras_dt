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
        active_rules.append((1, i + 1, rule))

    for i in range(2,len(w)):
        #print('i',i)
        for j in range(0,len(w)-i+2):
            #print('j',j)

            rule = P[j, i]

            if rule:
                active_rules.append((i+1-j, j+1, rule))
    return active_rules
#check if active rules are in P_dist
def checkInDist(active_rules, P_dist):
    for rules in active_rules:
        i = str(rules[0])
        j = str(rules[1])
        rule_l = rules[2]
        head = rule_l[0].rule.head()
        print(i,j,head)
        print(invsc(v(head)).dot(invsc(v(j))).dot(invsc(v(i))).dot(P_dist))


file = 'gramm_l'
w = 'a a b'
G = Grammar('S')
G.add_rules_from_file(file)

P = getP(w, G)
print(P)
P_dist = getPDistributed(w, G)

active_rules = getRules(w.replace(' ',''),P)

print(active_rules)
checkInDist(active_rules, P_dist)