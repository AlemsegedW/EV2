import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import sys


def print_solution(sol, var, I, K, T, L):
    x_val = sol.get_value_dict(var["x"])
    y_val = sol.get_value_dict(var["y"])
    p_val = sol.get_value_dict(var["p"])
    z_val = sol.get_value_dict(var["z"])
    # v_val = sol.get_value_dict(v)

    [print("x_{0}_{1}_{2} = %d".format(i, k, t) % x_val[i, k, t]) for i in I for k in K for t in T if
     x_val[i, k, t] != 0]
    [print("y_{0}=%d".format(k) % y_val[k]) for k in K if y_val[k] != 0]
    [print("z_{0}=%d".format(k) % z_val[k]) for k in K if z_val[k] != 0]
    [print("p_{0}_{1} = %f".format(k, t) % p_val[k, t]) for k in K for t in T if p_val[k, t] != 0]
    return None


def plot_solution(sol, var, vp, x_n, y_n, x_m, y_m, n, m, r, s, I, K, T, L, fig_name, problem):
    x_val = sol.get_value_dict(var["x"])
    y_val = sol.get_value_dict(var["y"])
    w_val = sol.get_value_dict(var["w"])
    if problem == "dynamic":
        v_val = sol.get_value_dict(var["v"])
    elif problem == "static":
        v_val = vp
    else:
        print("Error: problem must be static or dynamic")
        sys.exit(1)

    plt.figure(figsize=(10, 5))
    # breakpoint()
    plt.scatter(x_n[1:], y_n[1:], c='b', marker='s', label="user")
    plt.scatter(x_m[1:], y_m[1:], c='r', marker='s', label="CS")
    for i in K:
        plt.annotate('$%d$' % i, (x_m[i] - 0.02, y_m[i] + 0.025))

    plt.legend(loc="best")

    plt.axis('equal')

    for i in I:
        v_it = np.zeros((2, s + 1))
        for l in L:
            vi = np.zeros((m + 1, r + 1))
            wi = np.zeros((m + 1, r + 1))
            for k in K:
                for t in T:
                    vi[k, t] = v_val[i, k, t, l]
                    wi[k, t] = w_val[i, k, t, l]
            print((i))
            print((vi[1:, 1:]))
            print((wi[1:, 1:]))
            tm = np.nonzero(wi)
            print(tm)
            if np.size(tm) != 0:
                v_it[0, l] = tm[0]
                v_it[1, l] = tm[1]
            else:
                v_it[0, l] = 0
                v_it[1, l] = 0

            # print(v_it)

        plt.annotate('$%d$' % i, (x_n[i] - .02, y_n[i] + 0.025))
        v1 = v_it[0, 1:]
        v2 = v_it[1, 1:]
        # plt.annotate(str(list(v1.astype(np.int))), (x_n[i] + 0.02, y_n[i] - 0.01))
        plt.annotate(str(list(v1.astype(np.int))) + '\n' + str(list(v2.astype(np.int))), (x_n[i] + 0.02, y_n[i] - 0.01))

    [plt.annotate("x_{0}_{1}_{2}".format(i, k, t), (x_n[i] - 0.002, y_n[i] - 0.04))
     for i in I for k in K for t in T if x_val[i, k, t] == 1]
    [plt.annotate("y_{0} = %d".format(k) % y_val[k], (x_m[k] - 0.002, y_m[k] - 0.04), c='r') for k in K if
     y_val[k] != 0]
    # plt.axis('equal')
    plt.savefig(fig_name, bbox_inches='tight')
    plt.show(block=False)
    plt.close(fig_name)
