import cplex
import docplex.mp.model as cpx
import numpy as np
import time
from pref_list import preflist
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def solve_test1_middle_level(n, m, r, s, f_cost, p_list, p_u, delta_k, beta, gamma_k, vp, ld, x_n, y_n, x_m, y_m, d,a,b):
    I = range(1, n + 1)
    K = range(1, m + 1)
    T = range(1, r + 1)
    L = range(1, s + 1)
    # d = np.zeros((n + 1, m + 1, r + 1))
    beta_i = np.zeros(n + 1)
    alpha = np.zeros(n + 1)
    for i in I:
        beta_i[i] =b
        alpha[i] = a
        # for k in K:
        # for t in T:
        #   d[i, k, t] = ld[i, k]

    delta = {k: delta_k for k in K}  # limit on the individual size of the charging stations
    gamma = {k: gamma_k for k in K}  # capacity of individual charging stations
    # c = {k: rnd.uniform(.6, 1) for k in K}  # installation cost of  individual charging stations
    c = {k: 1 for k in K}  # installation cost of  individual charging stations
    # print(c)
    # v = {(i, k, t, l): vp[i, k, t, l] for i in I for k in K for t in T for l in L}  # preference list
    l_d = {(i, k): ld[i, k] * 1 for i in I for k in K}  # lambda of the follower problem

    # big M's
    M = 1000
    M5 = 1000
    M6 = 1000
    M1 = 1000
    M2 = 1000
    M3 = 1000
    prob = cpx.Model(name="EV Model")

    # primal leader and follower variables
    x = {(i, k, t): prob.binary_var(name="x_{0}_{1}_{2}".format(i, k, t)) for i in I for k in K for t in T}
    y = {k: prob.integer_var(lb=0, ub=np.inf, name="y_{0}".format(k)) for k in K}
    z = {k: prob.binary_var(name="z_{0}".format(k)) for k in K}
    p = {(k, t): prob.continuous_var(lb=0, ub=p_u, name="p_{0}_{1}".format(k, t)) for k in K for t in T}

    v = {(i, k, t, l): prob.binary_var(name="v_{0}_{1}_{2}_{3}".format(i, k, t, l)) for i in I for k in K for t in T for
         l in L}  # preference list
    eta = {(i, l): prob.continuous_var(lb=-np.inf, ub=None, name="eta_{0}_{1}".format(i, l)) for i in I for l in L}

    # dual follower variables
    pi = {(k, t): prob.continuous_var(lb=0, ub=None, name="pi_{0}_{1}".format(k, t)) for k in K for t in T}
    phi = {(i, k, t): prob.continuous_var(lb=0, ub=None, name="phi_{0}_{1}_{2}".format(i, k, t))
           for i in I for k in K for t in T}
    rho = {i: prob.continuous_var(lb=-np.inf, ub=None, name="rho_{0}".format(i)) for i in I}
    u = {(k, t): prob.binary_var(name="u_{0}_{1}".format(k, t)) for k in K for t in T}

    # leader constraints
    con71 = {
        k: prob.add_constraint(ct=z[k] - prob.sum(x[i, k, t] for i in I for t in T) <= 0, ctname="con71_{0}".format(k))
        for k in K}
    con72 = {(k, t): prob.add_constraint(ct=p[k, t] - prob.sum(x[i, k, t] for i in I for t in T) * p_u <= 0, ctname="con72_{0}_{1}".format(k, t))
             for k in K for t in T}
    con7b = {k: prob.add_constraint(ct=y[k] - delta[k] * z[k] <= 0, ctname="con7b_{0}".format(k)) for k in K}
    con7c = prob.add_constraint(ct=prob.sum(c[k] * y[k] for k in K) <= beta, ctname="con7c")

    # follower KKT
    con7f = {(i, k, t): prob.add_constraint(
        ct=f_cost * p[k, t] + pi[k, t] + rho[i] + phi[
            i, k, t] >= - l_d[i, k] - prob.sum(l * p_list * v[i, k, t, l] for l in L),
        ctname="con7f_{0}_{1}_{2}".format(i, k, t)) for i in I for k in K for t in T}
    # [print("v_{0}_{1}_{2} =  %d, con7f_{0}_{1}_{2} = %s".format(i, k, t,i,k,t) % (sum(l * p_list  * vp[i, k, t, l] for l in L), str(con7f[i,k,t]))) for i in I for k in K for t in T]
    con7g = {(i, k, t): prob.add_constraint(
        ct=f_cost * p[k, t] + l_d[i, k] + pi[k, t] + rho[i] + phi[
            i, k, t] + M * x[i, k, t] <= - prob.sum(l * p_list * v[i, k, t, l] for l in L) + M,
        ctname="con7g_{0}_{1}_{2}".format(i, k, t)) for i in I for k in K for t in T}

    con7h = {(k, t): prob.add_constraint(
        ct=prob.sum(x[i, k, t] for i in I) - gamma[k] * y[k] <= 0,
        ctname="con7h_{0}_{1}".format(k, t)) for k in K for t in T}

    con7i = {(k, t): prob.add_constraint(
        ct=gamma[k] * y[k] - prob.sum(x[i, k, t] for i in I) + u[k, t] * M5 <= M5,
        ctname="con7i_{0}_{1}".format(k, t)) for k in K for t in T}

    con7j = {(k, t): prob.add_constraint(
        ct=pi[k, t] - u[k, t] * M5 <= 0,
        ctname="con7j_{0}_{1}".format(k, t)) for k in K for t in T}

    con7k = {i: prob.add_constraint(
        ct=prob.sum(x[i, k, t] for k in K for t in T) == 1,
        ctname="con7k_{0}".format(i)) for i in I}

    con7l = {(i, k, t): prob.add_constraint(
        ct=x[i, k, t] - prob.sum(v[i, k, t, l] for l in L) <= 0,
        ctname="con7l_{0}_{1}_{2}".format(i, k, t)) for i in I for k in K for t in T}

    con7m = {(i, k, t): prob.add_constraint(
        ct=phi[i, k, t] + M6 * x[i, k, t] <= M6 * (1 + prob.sum(v[i, k, t, l] for l in L)),
        ctname="con7m_{0}_{1}_{2}".format(i, k, t)) for i in I for k in K for t in T}

    con6k = {(i, l): prob.add_constraint(ct=prob.sum(v[i, k, t, l] for k in K for t in T) == 1,
                                         ctname="con6k_{0}_{1}".format(i, l)) for i in I for l in L}

    con6f1 = {(i, k, t, 1): prob.add_constraint(ct=alpha[i] * p[k, t] + beta_i[i] * d[i, k, t] + eta[i, 1] >= 0,
                                                ctname="con6f1_{0}_{1}_{2}_{3}".format(i, k, t, 1)) for i in I for k in
              K for t in T}

    con6h1 = {(i, k, t, 1): prob.add_constraint(
        ct=alpha[i] * p[k, t] + beta_i[i] * d[i, k, t] + eta[i, 1] + M1 * v[i, k, t, 1] <= M1,
        ctname="con6h1_{0}_{1}_{2}_{3}".format(i, k, t, 1)) for i in I for k in K for t in T}

    if s > 1:
        xi = {(i, k, t, l, lp): prob.continuous_var(lb=0, ub=None, name="xi_{0}_{1}_{2}_{3}_{4}".format(i, k, t, l, lp))
              for i in I for k in K for t in T for l in range(2, s + 1) for lp in range(1, l)}

        con6f2 = {(i, k, t, l): prob.add_constraint(ct=alpha[i] * p[k, t] + beta_i[i] * d[i, k, t] + eta[i, l] +
                                                       prob.sum(xi[i, k, t, l, lp] for lp in range(1, l)) >= 0,
                                                    ctname="con6f2_{0}_{1}_{2}_{3}".format(i, k, t, l)) for i in I for k
                  in K for t in T for l in range(2, s + 1)}
        con6h2 = {(i, k, t, l): prob.add_constraint(
            ct=alpha[i] * p[k, t] + beta_i[i] * d[i, k, t] + eta[i, l] + prob.sum(
                xi[i, k, t, l, lp] for lp in range(1, l)) +
               M1 * v[i, k, t, l] <= M1, ctname="con6h2_{0}_{1}_{2}_{3}".format(i, k, t, l)) for i in I for k in K for t
            in T for l in range(2, s + 1)}

        con6j = {
            (i, k, t, l, lp): prob.add_constraint(xi[i, k, t, l, lp] - M3 * (v[i, k, t, l] + v[i, k, t, lp]) <= 0
                                                  , ctname="con6j_{0}_{1}_{2}_{3}_{4}".format(i, k, t, l, lp)) for i
            in I for k in K for t in T for l in range(2, s + 1) for lp in range(1, l)}

        con6l = {(i, k, t, l, lp): prob.add_constraint(v[i, k, t, l] + v[i, k, t, lp] <= 1
                                                       , ctname="con6l_{0}_{1}_{2}_{3}_{4}".format(i, k, t, l, lp)) for
                 i in I for k in K for t in T for l in range(2, s + 1) for lp in range(1, l)}

    objective = prob.sum(x[i, k, t] * p[k, t] for i in I for k in K for t in T) - 1 * prob.sum(
        x[i, k, t] for i in I for k in K for t in T) - 0 * prob.sum(c[k] * y[k] for k in K)

    prob.maximize(objective)
    prob.print_information()
    # breakpoint()
    prob.parameters.mip.tolerances.mipgap = 0.001
    prob.parameters.timelimit = 500
    sol = prob.solve(log_output=True)
    print(prob.get_solve_status())
    # prob.print_solution()
    print(sol.get_objective_value())
    x_val = sol.get_value_dict(x)
    y_val = sol.get_value_dict(y)
    p_val = sol.get_value_dict(p)
    z_val = sol.get_value_dict(z)
    v_val = sol.get_value_dict(v)


    [print("x_{0}_{1}_{2} = %d".format(i, k, t) % x_val[i, k, t]) for i in I for k in K for t in T if
     x_val[i, k, t] != 0]
    [print("y_{0}=%d".format(k) % y_val[k]) for k in K if y_val[k] != 0]
    [print("z_{0}=%d".format(k) % z_val[k]) for k in K if z_val[k] != 0]
    [print("p_{0}_{1} = %f".format(k, t) % p_val[k, t]) for k in K for t in T if p_val[k, t] != 0]
    plt.figure(figsize=(10, 5))
    # breakpoint()
    plt.scatter(x_n[1:], y_n[1:], c='b', marker='s', label="user")
    plt.scatter(x_m[1:], y_m[1:], c='r', marker='s', label="CS")
    for i in K:
        plt.annotate('$%d$' % i, (x_m[i] - 0.02, y_m[i] + 0.025))

    plt.legend(loc="best")

    plt.axis('equal')

    for i in I:
        v_it = np.zeros((2, s+1))
        for l in L:
            vi = np.zeros((m+1,r+1))
            for k in K:
                for t in T:
                    vi[k,t] = v_val[i, k, t, l]
            print((i))
            print((vi[1:,1:]))
            tm = np.nonzero(vi)
            v_it[0, l] = tm[0]
            v_it[1, l] = tm[1]
            #print(v_it)


        plt.annotate('$%d$' % i, (x_n[i] - .02, y_n[i] + 0.025))
        v1 = v_it[ 0,1:]
        v2= v_it[1, 1:]
        #plt.annotate(str(list(v1.astype(np.int))), (x_n[i] + 0.02, y_n[i] - 0.01))
        plt.annotate(str(list(v1.astype(np.int))) + '\n' + str(list(v2.astype(np.int))), (x_n[i] + 0.02, y_n[i] - 0.01))





    #breakpoint()

    # print(s)


    # plt.figure(figsize=(10, 5))
    # img = mpimg.imread('user_cs.png')


    [plt.annotate("x_{0}_{1}_{2}".format(i, k, t), (x_n[i] - 0.002, y_n[i] - 0.04))
     for i in I for k in K for t in T if x_val[i, k, t] == 1]
    [plt.annotate("y_{0} = %d".format(k) % y[k], (x_m[k] - 0.002, y_m[k] - 0.04), c='r') for k in K if y_val[k] != 0]
    # plt.axis('equal')
    plt.savefig('user_cs_3level', bbox_inches='tight')
    plt.show(block=False)
    plt.close('user_cs_3level')
    return None
