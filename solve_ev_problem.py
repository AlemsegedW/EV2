import cplex
import docplex.mp.model as cpx
import numpy as np
import time
from generate_problem_data import problem_data
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from ev_problem_static import solve_problem_static
from ev_problem_dynamic import solve_problem_dynamic
from ev_problem_static_w import solve_problem_static_w
import postprocess_solution

# imgplot = plt.imshow(img)
# plt.show()
# import matplotlib.pyplot as plt

# follower_cost
f_cost = 0

# follower_list
p_list = 1
# upper bound on p

p_u = 100
# rnd = np.random
# problem data
n = 10  # number of users
m = 5  # number of charging station candidates
r = 3  # number of time slats
s = 3  # number of preflist
delta_k = 2.0  # limit on the individual size of the charging stations
beta = 5.0 # Budget
gamma_k = 2.0

I = range(1, n + 1)  # set of indices of users
K = range(1, m + 1)  # set of indices of charging station candidates
T = range(1, r + 1)  # set of indices of time slats
L = range(1, s + 1)  # set of indices of preference list

print("delta_k " + str(delta_k) + ", beta = " + str(beta) + ", gamma_k " + str(gamma_k))
# generate sets


vp, ld, x_n, y_n, x_m, y_m, d = problem_data(n, m, r, s)
# sol, var = solve_problem_static(I, K, T, L, f_cost, p_list, p_u, delta_k, beta, gamma_k, vp, ld, x_n, y_n, x_m, y_m)
# postprocess_solution.print_solution(sol, var, I, K, T, L)
# postprocess_solution.plot_solution(sol, var, vp, x_n, y_n, x_m, y_m, n, m, r, s, I, K, T, L, "user_cs_static", "static")
# time.sleep(4)

sol, var = solve_problem_static_w(I, K, T, L, f_cost, p_list, p_u, delta_k, beta, gamma_k, vp, ld, s, n, r)
postprocess_solution.print_solution(sol, var, I, K, T, L)
postprocess_solution.plot_solution(sol, var, vp, x_n, y_n, x_m, y_m, n, m, r, s, I, K, T, L, "user_cs_static_w", "static")
time.sleep(4)

breakpoint()

# solve_test2(n, m, r, s, f_cost, p_list, p_u, delta_k, beta, gamma_k, vp, ld, x_n, y_n, x_m, y_m)
a = 0
b = 1
sol, var = solve_problem_dynamic(n, m, r, s, f_cost, p_list, p_u, delta_k, beta, gamma_k, vp, ld, x_n, y_n, x_m, y_m, d, a, b)
postprocess_solution.print_solution(sol, var, I, K, T, L)
postprocess_solution.plot_solution(sol, var, vp, x_n, y_n, x_m, y_m, n, m, r, s, I, K, T, L, "user_cs_dynamic", "dynamic")
time.sleep(4)

a = 1 / 2
b = 1 / 2
sol, var = solve_problem_dynamic(n, m, r, s, f_cost, p_list, p_u, delta_k, beta, gamma_k, vp, ld, x_n, y_n, x_m, y_m, d, a, b)
postprocess_solution.print_solution(sol, var, I, K, T, L)
postprocess_solution.plot_solution(sol, var, vp, x_n, y_n, x_m, y_m, n, m, r, s, I, K, T, L, "user_cs_dynamic_2", "dynamic")
