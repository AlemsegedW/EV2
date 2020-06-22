import cplex
import docplex.mp.model as cpx
import numpy as np
import time
from get_ev_problem_instances import ev_problem_instances
import matplotlib.pyplot as plt
import postprocess_solution

r_i = ev_problem_instances("n10-m5-M-5.txt")

prob = cpx.Model(name="EV Model")  # model name

# big M's
M = 1000.0
M5 = 1000.0
M6 = 1000.0
M0 = 1000.0

# extract rand_instances
data = ["s", "beta", "I", "K", "K_p", "T", "L", "delta", "gamma", "costy", "energy_cost", "lambda_ikt", "p_it", "vp"]
[print(r_i[data[i]]) for i in range(len(data))]
[s, beta, I, K, K_p, T, L, delta, gamma, costy, energe_cost, lambda_ikt, p_it, v] = [r_i["s"], r_i["beta"], r_i["I"],
                                                                                     r_i["K"], r_i["K_p"], r_i["T"],
                                                                                     r_i["L"], r_i["delta"],
                                                                                     r_i["gamma"], r_i["costy"],
                                                                                     r_i["energy_cost"],
                                                                                     r_i["lambda_ikt"], r_i["p_it"],
                                                                                     r_i["vp"]]

# primal leader and follower variables
x = {(i, k, t): prob.binary_var(name="x_{0}_{1}_{2}".format(i, k, t)) for i in I for k in K for t in T}
y = {k: prob.integer_var(lb=0, ub=None, name="y_{0}".format(k)) for k in K}
p = {(k, t): prob.continuous_var(lb=0, ub=None, name="p_{0}_{1}".format(k, t)) for k in K for t in T}
w = {(i, k, t, l): prob.binary_var(name="w_{0}_{1}_{2}_{3}".format(i, k, t, l)) for i in I for k in K for t in T for
     l in L}  # preference list
z = {i: prob.binary_var(name="z_{0}".format(i)) for i in I}

# dual follower variables
pi = {(k, t): prob.continuous_var(lb=0, ub=None, name="pi_{0}_{1}".format(k, t)) for k in K for t in T}
phi = {(i, k, t): prob.continuous_var(lb=0, ub=None, name="phi_{0}_{1}_{2}".format(i, k, t))
       for i in I for k in K for t in T}
rho = {i: prob.continuous_var(lb=-np.inf, ub=None, name="rho_{0}".format(i)) for i in I}
u = {(k, t): prob.binary_var(name="u_{0}_{1}".format(k, t)) for k in K for t in T}

con2b = {k: prob.add_constraint(ct=y[k] - delta[k] <= 0,
                                ctname="con7b_{0}".format(k))
         for k in K}
# [print("%s" % str(con2b[k])) for k in K]
con2c = prob.add_constraint(ct=prob.sum(costy[k] * y[k] for k in K) - beta <= 0,
                            ctname="con2c")
# [print("%s" % str(con2c))]

con2f = {(i, k, t, l): prob.add_constraint(ct=w[i, k, t, l] - v[i, k, t, l] <= 0,
                                           ctname="con2f_{0}_{1}_{2}_{3}".format(i, k, t, l))
         for i in I for k in K for t in T for l in L}
# [print("%s" % str(con2f[i,k,t,l])) for i in I for k in K for t in T for l in L]

con2g = {(i, k, t, l): prob.add_constraint(ct=w[i, k, t, l] - v[i, k, t, l] * (p_it[i, t] - p[k, t]) / M0 >= 0,
                                           ctname="con2g_{0}_{1}_{2}_{3}".format(i, k, t, l))
         for i in I for k in K for t in T for l in L}
# [print("%s" % str(con2g[i,k,t,l])) for i in I for k in K for t in T for l in L]

con2h = {(i, k, t, l): prob.add_constraint(ct=v[i, k, t, l] - w[i, k, t, l] -
                                              M0 * v[i, k, t, l] * prob.abs(p_it[i, t] - p[k, t]) <= 0,
                                           ctname="con2h_{0}_{1}_{2}_{3}".format(i, k, t, l))
         for i in I for k in K for t in T for l in L}
# [print("%s" % str(con2h[i,k,t,l])) for i in I for k in K for t in T for l in L]

con2i = {(i, k, t, l): prob.add_constraint(ct=w[i, k, t, l] - 1 -
                                              v[i, k, t, l] * (p_it[i, t] - p[k, t]) / M0 <= 0,
                                           ctname="con2i_{0}_{1}_{2}_{3}".format(i, k, t, l))
         for i in I for k in K for t in T for l in L}
# [print("%s" % str(con2i[i,k,t,l])) for i in I for k in K for t in T for l in L]

con2k = {i: prob.add_constraint(ct=z[i] - prob.sum(w[i, k, t, l] for k in K for t in T for l in L) <= 0,
                                ctname="con2k_{0}".format(i))
         for i in I}
# [print("%s" % str(con2k[i])) for i in I]

con2l = {i: prob.add_constraint(ct=z[i] - prob.sum(w[i, k, t, l] for k in K for t in T for l in L) / s >= 0,
                                ctname="con2l_{0}".format(i))
         for i in I}
# [print("%s" % str(con2l[i])) for i in I]

con2n = {(i, k, t): prob.add_constraint(ct=pi[k, t] + rho[i] + phi[i, k, t] + lambda_ikt[i, k, t] +
                                           prob.sum(l * w[i, k, t, l] for l in L) >= 0,
                                        ctname="con2n_{0}_{1}_{2}".format(i, k, t))
         for i in I for k in K for t in T}
# [print("%s" % str(con2n[i, k, t])) for i in I for k in K for t in T]

con20 = {(i, k, t): prob.add_constraint(ct=pi[k, t] + rho[i] + phi[i, k, t] + lambda_ikt[i, k, t] +
                                           prob.sum(l * w[i, k, t, l] for l in L) <= M - M * x[i,k,t],
                                        ctname="con20_{0}_{1}_{2}".format(i, k, t))
         for i in I for k in K for t in T}
[print("%s" % str(con20[i, k, t])) for i in I for k in K for t in T]