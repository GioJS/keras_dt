from cyk_dist import *
from functools import reduce

first = index0.dot(reduce(np.dot,[index1 for _ in range(10)]))
second = index0.dot(reduce(np.dot,[index1 for _ in range(10)]))

print(first.T.dot(second))

third = first.dot(second)
print(third)
print(third.T.dot(third))