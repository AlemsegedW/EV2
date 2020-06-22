# import cplex
# import docplex.mp.model as cpx
import numpy as np
import matplotlib.pyplot as plt

rnd = np.random


def problem_data(n, m, r, s):
    # global v
    vp = np.zeros((n + 1, m + 1, r + 1, s + 1))
    ld = np.zeros((n + 1, m + 1))  # distance b/n user i and CS k
    d = np.zeros((n + 1, m + 1, r + 1))  # distance b/n user i and CS k at time tr
    N = [i for i in range(1, n + 1)]
    M = [i for i in range(1, m + 1)]
    x_n = rnd.rand(len(N) + 1) * 2
    y_n = rnd.rand(len(N) + 1) * 1
    x_m = rnd.rand(len(M) + 1) * 2
    y_m = rnd.rand(len(M) + 1) * 1
    # for i in N:
    #   for j in M:
    #      plt.plot([x_n[i], x_m[j]], [y_n[i], y_m[j]], c='k')
    # plt.show()
    v = np.zeros((n + 1, s + 1))
    for i in N:
        ve = np.zeros((m + 1))
        for j in M:
            dist = np.sqrt((x_n[i] - x_m[j]) ** 2 + (y_n[i] - y_m[j]) ** 2)
            ve[j] = dist
            ld[i, j] = dist
            for t in range(1, r + 1):
                d[i, j, t] = dist + 1  # the plus 1 will be subtracted for those needed to be in the prefernce list
        ve = np.argsort(ve)
        tr = np.random.randint(1, r + 1, s)
        dp_i = 0
        #for j in ve[:s]:

            #for t in tr:
          #  d[i, j, tr[dp_i]] = d[i, j, tr[dp_i]] - 1 # - 1  to ensure the closest ones are in the preference list
           # dp_i = dp_i + 1
        v[i, :] = ve[:s + 1]

        for l in range(1, s + 1):
            # if l == 1:
            #    vp[i, int(v[i, l]), 1, l] = 1
            # else:
            #    vp[i, int(v[i, l]), rnd.randint(2, r + 1, 1), l] = 1
            # tr = rnd.randint(1, r + 1, 1)
            vp[i, int(v[i, l]), tr[l - 1], l] = 1
            d[i, int(v[i, l]), tr[l - 1]] = d[i, int(v[i, l]), tr[l - 1]]-1
    # for i in N:# print(vp)
    #    for l in range(1, s + 1):
    #         print(i)
    #        print(l*np.sum(vp[i, k, t, l] for k in M for t in range(1,r+1)))
    #        print(vp[i, 1:, 1:, l])
    if True:  # __name__ == "__main__":
        plt.figure(figsize=(10, 5))
        plt.scatter(x_n[1:], y_n[1:], c='b', marker='s', label="user")
        for i in N:
            plt.annotate('$%d$' % i, (x_n[i] - .02, y_n[i] + 0.025))
            tm = {l: np.argwhere(vp[i, :, :, l] == 1)[0, 1] for l in range(1, s + 1)}
            v1 = v[i, 1:]
            plt.annotate(str(list(v1.astype(np.int))) + '\n' + str(list(tm.values())), (x_n[i] + 0.02, y_n[i] - 0.01))
        plt.scatter(x_m[1:], y_m[1:], c='r', marker='s', label="CS")
        for i in M:
            plt.annotate('$%d$' % i, (x_m[i] - 0.02, y_m[i] + 0.025))

        plt.legend(loc="best")

        plt.axis('equal')
        plt.savefig('user_cs', bbox_inches='tight')
        # plt.show(block=False)
        # plt.ioff()
        # plt.show()
    print(ld)
    ld = ld / np.max(ld)
    d = d/1
    return vp, ld, x_n, y_n, x_m, y_m, d


if __name__ == "__main__":
    problem_data(10, 5, 3, 3)
