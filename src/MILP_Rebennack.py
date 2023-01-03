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
# c_max = 12
# c_min = 2
for i, j in cross_prod:
    c_trial = (data[i, 1] - data[j, 1]) / (data[i, 0] - data[j, 0])
    if c_trial > c_max:
        c_max = c_trial
    if c_trial < c_min:
        c_min = c_trial
        
print(c_min)
print(c_max)
# dmax: -13441, dmin: -81526
# Distance metric
q = 1