import itertools

import gurobipy as gp
import numpy as np



file = "data/MpStorage50.txt"
data = np.genfromtxt(file, delimiter="\t")

n_data_points = data.shape[0]
n_breakpoints = 5
# Distance metric
q = 1

cross_prod = itertools.product(range(n_data_points), repeat=2)
c_max = -float("inf")
c_min = float("inf")
for i, j in cross_prod:
    c_trial = (data[i, 1] - data[j, 1]) / (data[i, 0] - data[j, 0])
    if c_trial > c_max:
        c_max = c_trial
    if c_trial < c_min:
        c_min = c_trial

d_c = np.c_[
    data[:,1] - c_max*data[:,0],
    data[:,1] - c_min*data[:,0],
]
d_max = np.max(d_c)
d_min = np.min(d_c)
M_2 = d_max - d_min - data[:,0]*(c_min - c_max)
M_a_arr = np.c_[
    d_c - d_min,
    d_c - d_max,
]
M_a = np.max(M_a_arr, axis=1)

m = gp.Model("MILP_Rebennack")
c = m.addMvar(n_breakpoints - 1, name="c", lb=c_min, ub=c_max)
d = m.addMVar(n_breakpoints - 1, name="d", lb=d_min, ub=d_max)
gamma = np.zeros(n_breakpoints - 2, name="gamma", vtype=gp.GRB.BINARY)
delta = m.addMVar(
    (n_data_points, n_breakpoints - 1),
    vtype=gp.GRB.BINARY,
    name="delta",
)
delta_plus = m.addMVar(
    (n_data_points - 1, n_breakpoints - 2),
    lb=0,
    ub=1,
    name="delta_plus",
)
delta_minus = m.addMVar(
    (n_data_points - 1, n_breakpoints - 2),
    lb=0,
    ub=1,
    name="delta_minus",
)
epsilon = m.addMVar(
    n_data_points,
    lb=0,
    name="epsilon",
)
m.addConstr(
    gp.quicksum(
        delta[i,b]
        for i in range(n_data_points) 
        for b in range(n_breakpoints - 1)
    ) == 1
)






