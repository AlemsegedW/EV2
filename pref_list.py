# import cplex
# import docplex.mp.model as cpx
import numpy as np
import matplotlib.pyplot as plt

rnd = np.random


def preflist(n, m, r, s):
    # global v
    vp = np.zeros((n + 1, m + 1, r + 1, s + 1))
    N = [i for i in range(1, n + 1)]
    M = [i for i in range(1, m + 1)]
    x_n = rnd.rand(len(N) + 1) * 2
    y_n = rnd.rand(len(N) + 1) * 1
    x_m = rnd.rand(len(M) + 1) * 2
    y_m = rnd.rand(len(M) + 1) * 1
    if __name__ =="__main__":
        plt.figure(1)
        plt.subplot(211)
        plt.scatter(x_n[1:], y_n[1:], c='b', marker='s')
        for i in N:
            plt.annotate('$%d$' % i, (x_n[i] + 0, y_n[i]))
        plt.scatter(x_m[1:], y_m[1:], c='r', marker='s')
        for i in M:
            plt.annotate('$%d$' % i, (x_m[i] + 0, y_m[i]))
    # for i in N:
    #    for j in M:
    #       plt.plot([x_n[i], x_m[j]], [y_n[i], y_m[j]], c='k')
    # plt.show()
    v = np.zeros((n + 1, m + 1))
    for i in N:
        for j in M:
            v[i, j] = np.sqrt((x_n[i] - x_m[j]) ** 2 + (y_n[i] - y_m[j]) ** 2)

        v[i, :] = np.argsort(v[i, :])
        for l in range(1, s + 1):
            if l == 1:
                vp[i, int(v[i, l]), 1, l] = 1
            else:
                vp[i, int(v[i, l]), rnd.randint(2, r, 1), l] = 1

        # print(v)
    #for l in range(1, s + 1):
     #   print(vp[1, :, :, l])
    if __name__ == "__main__":
        plt.subplot(212)
        plt.scatter(x_n[1:], y_n[1:], c='b', marker='s')
        for i in N:
            plt.annotate(str(v[i, 1:]), (x_n[i] + 0, y_n[i]))
        plt.scatter(x_m[1:], y_m[1:], c='r', marker='s')
        for i in M:
            plt.annotate('$%d$' % i, (x_m[i] + 0, y_m[i]))
        plt.show()
        plt.axis('equal')
    return vp

preflist(10, 5, 4, 5)
