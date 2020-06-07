import cplex
import docplex.mp.model as cpx
import numpy as np
import time
from pref_list import preflist
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from EV_test1 import solve_test1
from EV_test2 import solve_test2
from EV_test1_middle_level import solve_test1_middle_level

# imgplot = plt.imshow(img)
# plt.show()
# import matplotlib.pyplot as plt

# follower_cost
f_cost = 0
# follower_list
p_list = 1
# upper bound on p
p_u = .5
rnd = np.random
# problem data
n = 10 # number of users
m =  8 # number of charging station candidates
r = 3 # number of time slats
s = 3  # number of preflist
delta_k = 2 # limit on the individual size of the charging stations
beta = np.ceil(delta_k * m / (2))  # Budget
gamma_k = max(np.ceil(2*n/ (r*delta_k*m)),1)
print("delta_k " + str(delta_k) + ", beta = " + str(beta) + ", gamma_k " + str(gamma_k))
# generate sets


vp, ld, x_n, y_n, x_m, y_m, d = preflist(n, m, r, s)
solve_test1(n, m, r, s, f_cost, p_list, p_u, delta_k, beta, gamma_k, vp, ld, x_n, y_n, x_m, y_m)
#time.sleep(4)
#breakpoint()
time.sleep(4)
#solve_test2(n, m, r, s, f_cost, p_list, p_u, delta_k, beta, gamma_k, vp, ld, x_n, y_n, x_m, y_m)
solve_test1_middle_level(n, m, r, s, f_cost, p_list, p_u, delta_k, beta, gamma_k, vp, ld, x_n, y_n, x_m, y_m, d)
