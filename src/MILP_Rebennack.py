import time
import itertools

import gurobipy as gp
import numpy as np

t = time.perf_counter()
file = "data/MpStorage50.txt"
data = np.genfromtxt(file, delimiter="\t")

data_points = data.shape[0]
n_breakpoints = 5

cross_prod = itertools.product(range(data_points), repeat=2)
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
print(d_max)
d_min = np.min(d_c)
print(d_min)
M_2 = d_max - d_min - data[:,0]*(c_min - c_max)
M_a_arr = np.c_[
    d_c - d_min,
    d_c - d_max,
]
M_a = np.max(M_a_arr, axis=1)
file_ma = "Ma.txt"
file_m2 = "m2.txt"
M2_test = np.genfromtxt(file_m2, delimiter="\n")
Ma_test = np.genfromtxt(file_ma, delimiter="\n")
print(all(M2_test == M_2))
print(all(Ma_test == M_a))
# Distance metric
q = 1