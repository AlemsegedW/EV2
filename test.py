import numpy as np
from heapq import nsmallest
ve = np.array([[100, 100, 100, 100],
                    [100, 2, 3,4],
                   [100, 4, 0, 5],
                   [100, 6, 2, 1],
                   [100, -1, 2, 1],
                   [100, 2, 5, 10]])
t1 = np.append([0*(np.array(range(4)))],np.ones((5,4))*np.array(range(4)) , axis=0).ravel()
t2 = np.zeros((6,4))
t2[1:,1:] = np.repeat(np.array(range(1,6)),3).reshape(5,3)
t2 = t2.ravel()
print(t1)
print(t2)
print(ve.ravel())


av = ve.ravel()
result = np.argsort(av)
print(result)
print("\nk smallest values:")
print(av[result[:3]])
v = np.zeros(4)

v[1:] = t2[result[:3]]
tm = t1[result[:3]]
print(v)
print(tm)