# import cplex
# import docplex.mp.model as cpx
import numpy as np
import matplotlib.pyplot as plt

rnd = np.random


def cs_user_coord(n, m):
    ld = np.zeros((n + 1, m + 1))
    I = [i for i in range(1, n + 1)]
    K = [i for i in range(1, m + 1)]
    x_n = rnd.rand(len(I) + 1) * 2
    y_n = rnd.rand(len(I) + 1) * 1
    x_m = rnd.rand(len(K) + 1) * 2
    y_m = rnd.rand(len(K) + 1) * 1
    for i in I:
        for j in K:
            dist = np.sqrt((x_n[i] - x_m[j]) ** 2 + (y_n[i] - y_m[j]) ** 2)
            ld[i, j] = dist
    if True:  # __name__ == "__main__":
        plt.figure(figsize=(10, 5))
        plt.scatter(x_n[1:], y_n[1:], c='b', marker='s', label="user")
        for i in I:
            plt.annotate('$%d$' % i, (x_n[i] - .02, y_n[i] + 0.025))
        plt.scatter(x_m[1:], y_m[1:], c='r', marker='s', label="CS")
        for i in K:
            plt.annotate('$%d$' % i, (x_m[i] - 0.02, y_m[i] + 0.025))
        plt.legend(loc="best")
        plt.axis('equal')
        plt.savefig('user_cs', bbox_inches='tight')
    ld = ld / np.max(ld)
    return ld, x_n, y_n, x_m, y_m


if __name__ == "__main__":
    cs_user_coord(10, 5)
