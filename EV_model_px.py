import cplex
import docplex.mp.model as cpx
import numpy as np
from pref_list import preflist
#import matplotlib.pyplot as plt


rnd = np.random
# problem data
n = 10  # number of users
m = 5  # number of charging station candidates
r = 3  # number of time slats
beta = m/2 # Budget

# generate sets
I = range(1, n + 1)
K = range(1, m + 1)
T = range(1, r + 1)
L = range(1, m + 1)

vp = preflist(n, m,r,m)
delta = {k: 10 for k in K}  # limit on the individual size of the charging stations
gamma = {k: 50 for k in K}  # capacity of individual charging stations
c = {k: rnd.uniform(0, 1) for k in K}  # installation cost of  individual charging stations
print(c)
v = {(i, k, t, l): vp[i,k,t,l] for i in I for k in K for t in T for l in L}  # preference list
l_d = {(i, k): rnd.uniform(0, 1)*1 for i in I for k in K}  # lambda of the follower problem

# big M's
M = 100000
M5 = 100000
M6 = 100000

prob = cpx.Model(name="EV Model")

# primal leader and follower variables
x = {(i, k, t): prob.binary_var(name="x_{0}_{1}_{2}".format(i, k, t)) for i in I for k in K for t in T}
y = {k: prob.integer_var(lb=0, ub=np.inf, name="y_{0}".format(k)) for k in K}
z = {k: prob.binary_var(name="z_{0}".format(k)) for k in K}
p = {(k, t): prob.continuous_var(lb=0, ub=None, name="p_{0}_{1}".format(k, t)) for k in K for t in T}

# dual follower variables
pi = {(k, t): prob.continuous_var(lb=0, ub=None, name="pi_{0}_{1}".format(k, t)) for k in K for t in T}
phi = {(i, k, t): prob.continuous_var(lb=0, ub=None, name="phi_{0}_{1}_{2}".format(i, k, t))
       for i in I for k in K for t in T}
rho = {i: prob.continuous_var(lb=-np.inf, ub=None, name="rho_{0}".format(i)) for i in I}
u = {(k, t): prob.binary_var(name="u_{0}_{1}".format(k, t)) for k in K for t in T}

# leader constraints
con7b = {k: prob.add_constraint(ct=y[k] - delta[k] * z[k] <= 0, ctname="con7b_{0}".format(k)) for k in K}
con7c = prob.add_constraint(ct=prob.sum(y[k] * c[k] for k in K) - beta <= 0, ctname="con7c")

# follower KKT
con7f = {(i, k, t): prob.add_constraint(
    ct=l_d[i, k] + prob.sum(l * v[i, k, t, l] for l in L) + pi[k, t] + rho[i] + phi[i, k, t] >= 0,
    ctname="con7f_{0}_{1}_{2}".format(i, k, t)) for i in I for k in K for t in T}

con7g = {(i, k, t): prob.add_constraint(
    ct=l_d[i, k] + prob.sum(l * v[i, k, t, l] for l in L) + pi[k, t] + rho[i] + phi[i, k, t] - M * (
            1 - x[i, k, t]) <= 0,
    ctname="con7g_{0}_{1}_{2}".format(i, k, t)) for i in I for k in K for t in T}

con7h = {(k, t): prob.add_constraint(
    ct=prob.sum(x[i, k, t] for i in I) - gamma[k] * y[k] <= 0,
    ctname="con7h_{0}_{1}".format(k, t)) for k in K for t in T}

con7i = {(k, t): prob.add_constraint(
    ct=gamma[k] * y[k] - prob.sum(x[i, k, t] for i in I) - (1 - u[k, t]) * M5 <= 0,
    ctname="con7i_{0}_{1}".format(k, t)) for k in K for t in T}

con7j = {(k, t): prob.add_constraint(
    ct=pi[k, t] - u[k, t] * M6 <= 0,
    ctname="con7j_{0}_{1}".format(k, t)) for k in K for t in T}

con7k = {i: prob.add_constraint(
    ct=prob.sum(x[i, k, t] for k in K for t in T) - 1 == 0,
    ctname="con7k_{0}".format(i)) for i in I}

con7l = {(i, k, t): prob.add_constraint(
    ct=x[i, k, t] - prob.sum(v[i, k, t, l] for l in L) <= 0,
    ctname="con7l_{0}_{1}_{2}".format(i, k, t)) for i in I for k in K for t in T}

con7m = {(i, k, t): prob.add_constraint(
    ct=phi[i, k, t] - M6 * (1 - x[i, k, t] + prob.sum(v[i, k, t, l] for l in L)) <= 0,
    ctname="con7m_{0}_{1}_{2}".format(i, k, t)) for i in I for k in K for t in T}

# objective function
objective = prob.sum(x[i, k, t] * p[k, t] for i in I for k in K for t in T) -prob.sum(
    x[i, k, t] for i in I for k in K for t in T) - prob.sum(
    y[k] for k in K)
prob.maximize(objective)
#breakpoint()
# solve the problem
# solve the problem
s = prob.solve()
s= prob.solve(log_output=True)
prob.print_information()
print(prob.get_solve_status())
print(s)