import cplex
import docplex.mp.model as cpx
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def solve_problem_static_w(I, K, T, L, f_cost, p_list, p_u, delta_k, beta, gamma_k, vp, ld, s, n, r):
    delta = {k: delta_k for k in K}  # limit on the individual size of the charging stations
    gamma = {k: gamma_k for k in K}  # capacity of individual charging stations
    # c = {k: rnd.uniform(.6, 1) for k in K}  # installation cost of  individual charging stations
    c = {k: 1.0 for k in K}  # installation cost of  individual charging stations
    v = {(i, k, t, l): vp[i, k, t, l] for i in I for k in K for t in T for l in L}  # preference list
    l_d = {(i, k): ld[i, k] for i in I for k in K}  # lambda of the follower problem
    p_it = {(i, t): 5.0 for i in I for t in T}  # price
    print(p_it)

    # big M's
    M = 1000.0
    M5 = 1000.0
    M6 = 1000.0

    prob = cpx.Model(name="EV Model")  # model name

    # primal leader and follower variables
    x = {(i, k, t): prob.binary_var(name="x_{0}_{1}_{2}".format(i, k, t)) for i in I for k in K for t in T}
    y = {k: prob.integer_var(lb=0, ub=np.inf, name="y_{0}".format(k)) for k in K}
    z = {i: prob.binary_var(name="z_{0}".format(i)) for i in I}
    p = {(k, t): prob.continuous_var(lb=0, ub=p_u, name="p_{0}_{1}".format(k, t)) for k in K for t in T}
    w = {(i, k, t, l): prob.binary_var(name="w_{0}_{1}_{2}_{3}".format(i, k, t, l)) for i in I for k in K for t in T for
         l in L}  # preference list

    # for i in I: # print v
    #  print(i)
    # for l in L:
    #     zd = {(i, k, t, l): v[i, k, t, l] for k in K for t in T}
    #    print(np.reshape(list(zd.values()), (m, r)))

    # dual follower variables
    pi = {(k, t): prob.continuous_var(lb=0, ub=None, name="pi_{0}_{1}".format(k, t)) for k in K for t in T}
    phi = {(i, k, t): prob.continuous_var(lb=0, ub=None, name="phi_{0}_{1}_{2}".format(i, k, t))
           for i in I for k in K for t in T}
    rho = {i: prob.continuous_var(lb=-np.inf, ub=None, name="rho_{0}".format(i)) for i in I}
    u = {(k, t): prob.binary_var(name="u_{0}_{1}".format(k, t)) for k in K for t in T}

    # leader constraints
    con72 = {(k, t): prob.add_constraint(ct=p[k, t] - prob.sum(x[i, k, t] for i in I) * p_u <= 0,
                                         ctname="con72_{0}_{1}".format(k, t)) for k in K for t in T}
    con6b = {k: prob.add_constraint(ct=y[k] - delta[k] <= 0, ctname="con6b_{0}".format(k)) for k in K}
    #con6b = {k: prob.add_constraint(ct=y[k] - delta[k]*prob.sum(x[i, k, t] for i in I for t in T) <= 0,
     #                               ctname="con6b_{0}".format(k)) for k in K}
    con6c = prob.add_constraint(ct=prob.sum(c[k] * y[k] for k in K) <= beta, ctname="con6c")
    con6f = {(i, k, t, l): prob.add_constraint(ct=w[i, k, t, l] - v[i, k, t, l] <= 0,
                                               ctname="con6f_{0}_{1}_{2}_{3}".format(i, k, t, l)) for i in I for k in K
             for t in T for l in L}
    # con6g = {(i, k, t, l): prob.add_constraint(ct=w[i, k, t, l] - 1/ p_it[i, t]*(v[i, k, t, l]*p_it[i, t] -p[k,t]) >= 0, ctname="con6g_{0}_{1}_{2}_{3}".format(i, k, t, l)) for i in I for k
    #        in K for t in T for l in L}
    con6h = {(i, k, t, l): prob.add_constraint(w[i, k, t, l] - 1.0 + 0.00000 - (v[i, k, t, l] * p_it[i, t] -
                                                                                p[k, t]) / p_u <= 0,
                                               ctname="con6h_{0}_{1}_{2}_{3}".format(i, k, t, l)) for i in I for k in K
             for t in T for l in L}
    con6j = {i: prob.add_constraint(ct=z[i] - prob.sum(w[i, k, t, l] for k in K for t in T for l in L) <= 0,
                                    ctname="con6j_{0}".format(i)) for i in I}
    con6k = {i: prob.add_constraint(ct=z[i] - prob.sum(w[i, k, t, l] for k in K for t in T for l in L) / s >= 0,
                                    ctname="con6k_{0}".format(i)) for i in I}
    # follower KKT
    con7f = {(i, k, t): prob.add_constraint(
        ct=f_cost * p[k, t] + pi[k, t] + rho[i] + phi[
            i, k, t] >= - l_d[i, k] - prob.sum(l * p_list * w[i, k, t, l] for l in L),
        ctname="con7f_{0}_{1}_{2}".format(i, k, t)) for i in I for k in K for t in T}
    # [print("v_{0}_{1}_{2} =  %d, con7f_{0}_{1}_{2} = %s".format(i, k, t,i,k,t) % (sum(l * p_list  * vp[i, k, t, l] for l in L), str(con7f[i,k,t]))) for i in I for k in K for t in T]
    con7g = {(i, k, t): prob.add_constraint(
        ct=f_cost * p[k, t] + pi[k, t] + rho[i] + phi[
            i, k, t] + M * x[i, k, t] <= - l_d[i, k] - prob.sum(l * p_list * w[i, k, t, l] for l in L) + M,
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
        ct=prob.sum(x[i, k, t] for k in K for t in T) - z[i] == 0,
        ctname="con7k_{0}".format(i)) for i in I}

    con7l = {(i, k, t): prob.add_constraint(
        ct=x[i, k, t] - prob.sum(w[i, k, t, l] for l in L) <= 0,
        ctname="con7l_{0}_{1}_{2}".format(i, k, t)) for i in I for k in K for t in T}

    con7m = {(i, k, t): prob.add_constraint(
        ct=phi[i, k, t] + M6 * x[i, k, t] <= M6 * (1 + prob.sum(w[i, k, t, l] for l in L)),
        ctname="con7m_{0}_{1}_{2}".format(i, k, t)) for i in I for k in K for t in T}

    # con7s = {(i, k, t): prob.add_constraint(
    # ct=w_linear[i, k, t] -w[i, k, t] ** 2 - 0 <= 0,
    # ctname="con7s_{0}_{1}_{2}".format(i, k, t)) for i in I for k in K for t in T}
    # objective function
    objective = prob.sum(x[i, k, t] * p[k, t] for i in I for k in K for t in T) - 1 * prob.sum(
        x[i, k, t] for i in I for k in K for t in T) - 1 * prob.sum(c[k] * y[k] for k in K)

    prob.maximize(objective)
    prob.print_information()

    # prob.parameters.mip.tolerances.mipgap = 0.001
    # prob.parameters.timelimit = 500
    sol_static = prob.solve(log_output=True)
    print(prob.get_solve_status())
    # prob.print_solution()
    print(sol_static.get_objective_value())
    var = {"x": x, "y": y, "z": z, "p": p, "w": w}

   # [print("w_{0}_{1}_{2}_{3} = %d,".format(i, k, t, l) % (w[i, k, t, l]), str(con6fg[i, k, t, l]))
    # for i in I for k in K
    # for t in T for l in L]

    #breakpoint()
    return sol_static, var
