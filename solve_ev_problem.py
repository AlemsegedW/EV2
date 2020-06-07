import cplex
import docplex.mp.model as cpx
import numpy as np
import time
from generate_problem_data import problem_data
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from ev_problem_static import solve_problem_static
from EV_test2 import solve_test2
from ev_problem_dynamic import solve_problem_dynamic

# imgplot = plt.imshow(img)
# plt.show()
# import matplotlib.pyplot as plt

# follower_cost
f_cost = 0

# follower_list
p_list = 1
# upper bound on p

p_u = 2
rnd = np.random
# problem data
n = 20 # number of users
m =  5 # number of charging station candidates
r = 3 # number of time slats
s = 3  # number of preflist
delta_k = 2 # limit on the individual size of the charging stations
beta = np.ceil(m)  # Budget
gamma_k = 2

print("delta_k " + str(delta_k) + ", beta = " + str(beta) + ", gamma_k " + str(gamma_k))
# generate sets


vp, ld, x_n, y_n, x_m, y_m, d =problem_data(n, m, r, s)
solve_problem_static(n, m, r, s, f_cost, p_list, p_u, delta_k, beta, gamma_k, vp, ld, x_n, y_n, x_m, y_m)
#time.sleep(4)
#breakpoint()
time.sleep(4)
#solve_test2(n, m, r, s, f_cost, p_list, p_u, delta_k, beta, gamma_k, vp, ld, x_n, y_n, x_m, y_m)
a = 0
b = 1
solve_problem_dynamic(n, m, r, s, f_cost, p_list, p_u, delta_k, beta, gamma_k, vp, ld, x_n, y_n, x_m, y_m, d, a, b)
time.sleep(4)
a = 1/2
b = 1/2
breakpoint()
solve_problem_dynamic(n, m, r, s, f_cost, p_list, p_u, delta_k, beta, gamma_k, vp, ld, x_n, y_n, x_m, y_m, d, a, b)
