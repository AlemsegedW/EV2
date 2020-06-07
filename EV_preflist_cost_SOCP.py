import cplex
import docplex.mp.model as cpx
import numpy as np

from pref_list import preflist
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# imgplot = plt.imshow(img)
# plt.show()
# import matplotlib.pyplot as plt

# follower_cost
f_cost = 0
# follower_list
p_list = 1
# upper bound on p
p_u = 10
rnd = np.random
# problem data
n = 10  # number of users
m = 8 # number of charging station candidates
r = 3  # number of time slats
s = 4  # number of preflist
delta_k = 5 # limit on the individual size of the charging stations
beta = np.ceil(delta_k*m/(2*r))  # Budget
gamma_k = np.ceil(delta_k/delta_k)
print("beta = %d, gamma = gamma_k =%d", beta, gamma_k)
# generate sets
I = range(1, n + 1)
K = range(1, m + 1)
T = range(1, r + 1)
L = range(1, s + 1)

vp, ld, x_n, y_n, x_m, y_m = preflist(n, m, r, s)
delta = {k: delta_k for k in K}  # limit on the individual size of the charging stations
gamma = {k: gamma_k for k in K}  # capacity of individual charging stations
# c = {k: rnd.uniform(.6, 1) for k in K}  # installation cost of  individual charging stations
c = {k: 1 for k in K}  # installation cost of  individual charging stations
# print(c)
v = {(i, k, t, l): vp[i, k, t, l] for i in I for k in K for t in T for l in L}  # preference list
l_d = {(i, k): ld[i, k] * 0 for i in I for k in K}  # lambda of the follower problem

# big M's
M = 100
M5 = 100
M6 = 100

prob = cpx.Model(name="EV Model")

# primal leader and follower variables
x = {(i, k, t): prob.binary_var(name="x_{0}_{1}_{2}".format(i, k, t)) for i in I for k in K for t in T}
# x = {(i, k, t): prob.continuous_var(lb=0, ub=1, name="x_{0}_{1}_{2}".format(i, k, t)) for i in I for k in K for t in T}
y = {k: prob.integer_var(lb=0, ub=np.inf, name="y_{0}".format(k)) for k in K}
z = {k: prob.binary_var(name="z_{0}".format(k)) for k in K}
p = {(k, t): prob.continuous_var(lb=0, ub=p_u, name="p_{0}_{1}".format(k, t)) for k in K for t in T}

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
con71 = {k: prob.add_constraint(ct=z[k] - prob.sum(x[i, k, t] for i in I for t in T) <= 0, ctname="con71_{0}".format(k))
         for k in K}
con72 = {(k, t): prob.add_constraint(ct=p[k, t] - z[k] * p_u <= 0, ctname="con72_{0}_{1}".format(k, t))
         for k in K for t in T}
con7b = {k: prob.add_constraint(ct=y[k] - delta[k] * z[k] <= 0, ctname="con7b_{0}".format(k)) for k in K}
con7c = prob.add_constraint(ct=prob.sum(c[k] * y[k] for k in K) - beta <= 0, ctname="con7c")

# follower KKT
con7f = {(i, k, t): prob.add_constraint(
    ct=f_cost * p[k, t] + l_d[i, k] + prob.sum(l * p_list * v[i, k, t, l] for l in L) + pi[k, t] + rho[i] + phi[
        i, k, t] >= 0,
    ctname="con7f_{0}_{1}_{2}".format(i, k, t)) for i in I for k in K for t in T}

con7g = {(i, k, t): prob.add_constraint(
    ct=f_cost * p[k, t] + l_d[i, k] + prob.sum(l * p_list * v[i, k, t, l] for l in L) + pi[k, t] + rho[i] + phi[
        i, k, t] - M * (
               1 - x[i, k, t]) <= 0,
    ctname="con7g_{0}_{1}_{2}".format(i, k, t)) for i in I for k in K for t in T}

con7h = {(k, t): prob.add_constraint(
    ct=prob.sum(x[i, k, t] for i in I) - gamma[k] * y[k] <= 0,
    ctname="con7h_{0}_{1}".format(k, t)) for k in K for t in T}

con7i = {(k, t): prob.add_constraint(
    ct=gamma[k] * y[k] - prob.sum(x[i, k, t] for i in I) - (1 - u[k, t]) * M5 <= 0,
    ctname="con7i_{0}_{1}".format(k, t)) for k in K for t in T}

con7j = {(k, t): prob.add_constraint(
    ct=pi[k, t] - u[k, t] * M5 <= 0,
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

# con7s = {(i, k, t): prob.add_constraint(
# ct=w_linear[i, k, t] -w[i, k, t] ** 2 - 0 <= 0,
# ctname="con7s_{0}_{1}_{2}".format(i, k, t)) for i in I for k in K for t in T}
# objective function
objective = prob.sum(x[i, k, t] * p[k, t] for i in I for k in K for t in T) - 0 * prob.sum(
    x[i, k, t] for i in I for k in K for t in T) - 0 * prob.sum(c[k] * y[k] for k in K)

prob.maximize(objective)
prob.print_information()
# breakpoint()
#prob.parameters.mip.tolerances.mipgap = 0.001
#prob.parameters.timelimit = 500
s = prob.solve(log_output=True)
print(prob.get_solve_status())
# prob.print_solution()
print(s.get_objective_value())
x_val = s.get_value_dict(x)
y_val = s.get_value_dict(y)
p_val = s.get_value_dict(p)
z_val = s.get_value_dict(z)
[print("x_{0}_{1}_{2} = %d".format(i, k, t) % x_val[i, k, t]) for i in I for k in K for t in T if x_val[i, k, t] != 0]
[print("y_{0}=%d".format(k) % y_val[k]) for k in K if y_val[k] != 0]
[print("z_{0}=%d".format(k) % z_val[k]) for k in K if z_val[k] != 0]
[print("p_{0}_{1} = %f".format(k, t) % p_val[k, t]) for k in K for t in T if p_val[k, t] != 0]

# print(s)


# plt.figure(figsize=(10, 5))
img = mpimg.imread('user_cs.png')
[plt.annotate("x_{0}_{1}_{2}".format(i, k, t), (x_n[i] - 0.002, y_n[i] - 0.04))
 for i in I for k in K for t in T if x_val[i, k, t] == 1]
[plt.annotate("y_{0} = %d".format(k) % y[k], (x_m[k] - 0.002, y_m[k] - 0.04), c='r') for k in K if y_val[k] != 0]
# plt.axis('equal')
plt.savefig('user_cs_PX')
plt.show(block=False)
# plt.close()
# plt.show()

'''
# this is for scocp
# con7r.clear()
# w.clear()
# objectiveSOCP.clear()
# prob.remove_constraints("con7r_{0}_{1}_{2}".format(i, k, t) for i in I for k in K for t in T)
w = {(i, k, t): prob.continuous_var(lb=-np.inf, ub=None, name="w_{0}_{1}_{2}".format(i, k, t)) for i in I for k in K for
     t in T}
# w_linear = {(i, k, t): prob.continuous_var(lb=0, ub=100, name="w_linear_{0}_{1}_{2}".format(i, k, t)) for i in I for k in K for t in T}

con7r = {(i, k, t): prob.add_constraint(
    ct=w[i, k, t] ** 2 - x[i, k, t] * p[k, t] <= 0,
    ctname="con7r_{0}_{1}_{2}".format(i, k, t)) for i in I for k in K for t in T}

objectiveSOCP = prob.sum(w[i, k, t] ** 1 for i in I for k in K for t in T) - 0 * prob.sum(
    x[i, k, t] for i in I for k in K for t in T) - 0*prob.sum(c[k] * y[k] for k in K)

prob.maximize(objectiveSOCP)
prob.print_information()
# breakpoint()
prob.parameters.mip.tolerances.mipgap = 0.05
prob.parameters.timelimit = 20
s = prob.solve(log_output=True)
print(prob.get_solve_status())
# prob.print_solution()
print(s.get_objective_value())
x_val = s.get_value_dict(x)
y_val = s.get_value_dict(y)
p_val = s.get_value_dict(p)
z_val = s.get_value_dict(z)
[print("x_{0}_{1}_{2} = %d".format(i, k, t) % x_val[i, k, t]) for i in I for k in K for t in T if x_val[i, k, t] != 0]
[print("y_{0}=%d".format(k) % y_val[k]) for k in K if y_val[k] != 0]
[print("z_{0}=%d".format(k) % z_val[k]) for k in K if z_val[k] != 0]
[print("p_{0}_{1} = %f".format(k, t) % p_val[k, t]) for k in K for t in T if p_val[k, t] != 0]

# print(s)


# plt.figure(figsize=(10, 5))
img = mpimg.imread('user_cs.png')
[plt.annotate("x_{0}_{1}_{2}".format(i, k, t), (x_n[i] - 0.02, y_n[i] - 0.4))
 for i in I for k in K for t in T if x_val[i, k, t] == 1]
[plt.annotate("y_{0} = %d".format(k) % y[k], (x_m[k] - 0.02, y_m[k] - 0.4), c='r') for k in K if y_val[k] != 0]
# plt.axis('equal')
plt.savefig('user_cs_SOCP')
plt.show(block=False)
'''
