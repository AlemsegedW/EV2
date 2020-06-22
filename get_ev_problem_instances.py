import numpy as np
import matplotlib.pyplot as plt

rnd = np.random


def ev_problem_instances(file_name):
    try:
        with open("random_instances/" + file_name) as f:
            data = f.read()
            lines = data.split("\n")
    except FileNotFoundError:
        print(file_name + "is not found")
    # get n m M budget
    line_no = lines.index("n    m    M    Budget")
    n_m_beta = list(lines[line_no + 1].split())
    n = int(n_m_beta[0])  # number of users
    m = int(n_m_beta[1])  # number of CSs
    r = 3  # number of time slats
    s = 3
    M = int(n_m_beta[2])  # number of CSs
    beta = float(n_m_beta[3])  # budget
    print("n    m    M    Budget")
    print(n)
    print(m)
    print(M)
    print(beta)
    I = range(1, n + 1)  # indices of users
    K = range(1, m + 1)  # indices of CSs
    T = range(1, r + 1)  # indices of time slats
    L = range(1, s + 1)  # indices of time slats
    K_p = range(1, M + 1)  # indices of private time slats

    # get user coordniates
    line_no = lines.index("Users coordinations")
    user_coordinates = list(lines[line_no + 1:line_no + n + 1])
    x_n = np.zeros((n + 1, r + 1))
    y_n = np.zeros((n + 1, r + 1))
    for i in I:
        xy_u = user_coordinates[i - 1].split()
        for t in T:
            x_n[i, t] = float(xy_u[2 * t - 2])  #
            y_n[i, t] = float(xy_u[2 * t - 1])

    print("\nUsers coordinations")
    print(x_n[1:, 1:])
    print(y_n[1:, 1:])

    # get CS coordniates
    line_no = lines.index("CS candidates coordinations")
    cs_coordinates = list(lines[line_no + 1:line_no + m + 1])
    x_m = np.zeros((m + 1))
    y_m = np.zeros((m + 1,))
    for k in K:
        xy_c = cs_coordinates[k - 1].split()
        x_m[k] = float(xy_c[0])
        y_m[k] = float(xy_c[1])

    print("\nCS candidates coordinations")
    print(x_m)
    print(y_m)

    # get private CS coordniates
    line_no = lines.index("Private CSs coordinations")
    private_cs_coordinates = list(lines[line_no + 1:line_no + M + 1])
    x_mp = np.zeros((M + 1))
    y_mp = np.zeros((M + 1,))
    for k in K_p:
        xy_cp = private_cs_coordinates[k - 1].split()
        x_mp[k] = float(xy_cp[0])
        y_mp[k] = float(xy_cp[1])

    print("\nPrivate CS candidates coordinations")
    print(x_mp)
    print(y_mp)

    # get delta  gamma  cost-y  energy-cost
    line_no = lines.index("delta  gamma  cost-y  energy-cost")
    delta_gamma_costy_energycost = list(lines[line_no + 1:line_no + m + 1])
    delta = np.zeros((m + 1))
    gamma = np.zeros((m + 1))
    costy = np.zeros((m + 1))
    energy_cost = np.zeros((m + 1))
    for k in K_p:
        delta_gamma_costy_energycost_k = delta_gamma_costy_energycost[k - 1].split()
        delta[k] = float(delta_gamma_costy_energycost_k[0])
        gamma[k] = float(delta_gamma_costy_energycost_k[1])
        costy[k] = float(delta_gamma_costy_energycost_k[2])
        energy_cost[k] = float(delta_gamma_costy_energycost_k[3])

    print("\ndelta  gamma  cost-y  energy-cost")
    print(str(delta) + str(gamma) + str(costy) + str(energy_cost))

    # get alpha
    line_no = lines.index("alpha")
    a_k = list(lines[line_no + 1:line_no + n + 1])
    alpha = np.zeros((n + 1))
    for i in I:
        alpha[i] = float(a_k[i - 1].split()[0])
    print("\nalpha")
    print(alpha)

    # get lambda[i,k,t]
    line_no = lines.index("lambda[i,k,t]")
    lam_ikt = list(lines[line_no + 1:line_no + r * n + 1])
    lambda_ikt = np.zeros((n + 1, m + 1, r + 1))
    for i in I:
        for t in T:
            lambda_ikt[i, 1:, t] = np.float_(lam_ikt[n * (t - 1) + i - 1].split())
    print("\nlambda[i,k,t]")
    for t in T:
        print(lambda_ikt[1:, 1:, t])

    # get P_i^t
    line_no = lines.index("P_i^t")
    pit = list(lines[line_no + 1:line_no + n + 1])
    p_it = np.zeros((n + 1, r + 1))
    for i in I:
        p_it[i, 1:] = np.float_(pit[i - 1].split())
    print("\nP_i^t")
    print(p_it)

    vp = np.zeros((n + 1, m + 1, r + 1, s + 1))
    d = np.zeros((n + 1, m + 1, r + 1))
    v = np.zeros((n + 1, s + 1))
    tr = np.zeros((s + 1))
    t1 = np.append([0 * (np.array(range(r + 1)))], np.ones((m, r + 1)) * np.array(range(r + 1)), axis=0).ravel()
    t2 = np.zeros((m + 1, r + 1))
    t2[1:, 1:] = np.repeat(np.array(range(1, m + 1)), r).reshape(m, r)
    t2 = t2.ravel()
    for i in I:
        ve = 1000 * np.ones((m + 1, r + 1))
        for k in K:
            for t in T:
                dist = np.sqrt((x_n[i, t] - x_m[k]) ** 2 + (y_n[i, t] - y_m[k]) ** 2)
                ve[k, t] = dist
        av = ve.ravel()
        result = np.argsort(av)
        print("\nk smallest values:")
        print(av[result[:s]])
        v[i, 1:] = t2[result[:s]]
        tr[1:] = t1[result[:s]]
        print(t1)
        print(t2)
        print(ve.ravel())
        print(ve)
        print(result)
        print(tr)
        print(v[i, 0:])
        for l in range(1, s + 1):
            vp[i, int(v[i, l]), int(tr[l]), l] = 1

    if True:  # __name__ == "__main__":
        plt.figure(figsize=(10, 10))
        color = ["b", "g", "y"]
        for t in T:
            plt.scatter(x_n[1:, t], y_n[1:, t], c=color[t - 1], marker='o', label="user_t={0}".format(t))

        for i in I:
            for t in T:
                plt.annotate('$%d$' % i, (x_n[i, t] - .1, y_n[i, t] + 0.1))

            tm = {l: np.argwhere(vp[i, :, :, l] == 1)[0, 1] for l in range(1, s + 1)}
            v1 = v[i, 1:]
            plt.annotate(str(list(v1.astype(np.int))) + '\n' + str(list(tm.values())) + '\n' + str(
                list(p_it[i, (list(tm.values()))].astype(np.int))),
                         (x_n[i, 1] + 0.1, y_n[i, 1] - 0.01))
        plt.scatter(x_m[1:], y_m[1:], c='r', marker='s', label="CS")
        for k in K:
            plt.annotate('$%d$' % k, (x_m[k] - 0.1, y_m[k] + 0.1))

        plt.legend(loc="best")

        plt.axis('equal')
        plt.savefig('user_cs', bbox_inches='tight')
        # plt.show(block=False)
        # plt.ioff()
        # plt.show()
        rand_instances = {"n": n, "m": m, "M": M, "beta": beta, "r": r, "s": s,
                          "user_co": [x_n, y_n],
                          "cs_co": [x_m, y_m],
                          "private_cs_co": [x_m, y_m],
                          "delta": delta, "gamma": gamma, "costy": costy, "energy_cost": energy_cost,
                          "lambda_ikt": lambda_ikt, "p_it": p_it,
                          "I": I, "K": K, "T": T, "K_p": K_p, "L": L, "vp": vp}
    return rand_instances


if __name__ == "__main__":
    ev_problem_instances("n10-m5-M-5.txt")
