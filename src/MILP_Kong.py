import itertools

import gurobipy as gp
import numpy as np


file = "data/MpStorage50.txt"
# It is assumed that the data for the independet variables ("x") is in the first column
# and the dependent variable ("y") is in the second column
data = np.genfromtxt(file, delimiter="\t")

n_data_points = data.shape[0]
linear_segments = 4
n_breakpoints = linear_segments + 1
# Choose distance metric: feasibility, LInf, L1, L2
distance_metric = "L1" 

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

m = gp.Model("MILP_Kong")

Y = m.addMVar((n_data_points, n_breakpoints - 1), name="Y")
A = m.addMVar(n_breakpoints - 1, name="A", lb=c_min, ub=c_max)
B = m.addMVar(n_breakpoints - 1, name="B", lb=d_min, ub=d_max)
E = m.addMVar(n_data_points, name="E", lb=0)
E1 = m.addVar(name="E1", lb=0)

P_plus = m.addMVar((n_data_points + 1, n_breakpoints - 1), name="P_plus", lb=0)
P_minus = m.addMVar((n_data_points + 1, n_breakpoints - 1), name="P_minus", lb=0)

Q_plus = m.addMVar((n_data_points + 1, n_breakpoints - 1), name="Q_plus", lb=0)
Q_minus = m.addMVar((n_data_points + 1, n_breakpoints - 1), name="Q_minus", lb=0)

U = m.addMVar((n_data_points + 1, n_breakpoints - 2), name="U")
V = m.addMVar((n_data_points + 1, n_breakpoints - 2), name="V")

Z = m.addMVar((n_data_points + 1, n_breakpoints - 1), name="Z")
ZF = m.addMVar((n_data_points + 1, n_breakpoints - 1), name="ZF")
ZL = m.addMVar((n_data_points + 1, n_breakpoints - 1), name="ZL")

# Sum over bZ
for i in range(n_data_points):
    m.addConstr(gp.quicksum(Z[i, :]) == 1)

if distance_metric == "L1":
    for i in range(n_data_points):
        for breakpoint in range(n_breakpoints -1):
            m.addConstr(
                E1 >= (data[i, 0] * A[breakpoint] + B[breakpoint]) - data[i, 1] - M_a[i] * (1 - Z[i, breakpoint])
            )
            m.addConstr(
                E1 >= data[i, 1] - (data[i, 0] * A[breakpoint] + B[breakpoint]) - M_a[i] * (1 - Z[i, breakpoint])
            )
else:
    for i in range(n_data_points):
        for breakpoint in range(n_breakpoints -1):
            m.addConstr(
                E[i] >= (data[i, 0] * A[breakpoint] + B[breakpoint]) - data[i, 1] - M_a[i] * (1 - Z[i, breakpoint])
            )
            m.addConstr(
                E[i] >= data[i, 1] - (data[i, 0] * A[breakpoint] + B[breakpoint]) - M_a[i] * (1 - Z[i, breakpoint])
            )

for breakpoint in range(n_breakpoints - 1):
    m.addConstr(Z[0, breakpoint] == ZF[0, breakpoint])
    for i in range(1, n_data_points):
        m.addConstr(
            Z[i, breakpoint] == Z[i-1, breakpoint] + ZF[i, breakpoint] - ZL[i-1, breakpoint]
        )

for breakpoint in range(n_breakpoints - 1):
    m.addConstr(gp.quicksum(ZL[:, breakpoint]) == 1)
    m.addConstr(gp.quicksum(ZF[:, breakpoint]) == 1)

# partsums
for i in range(n_data_points - 1):
    for breakpoint in range(n_breakpoints - 2):
        m.addConstr(gp.quicksum(ZF[:i+1, breakpoint+1]) <= gp.quicksum(ZF[:i+1, breakpoint]))
        m.addConstr(gp.quicksum(ZL[:i+1, breakpoint+1]) <= gp.quicksum(ZL[:i+1, breakpoint]))
        

for i in range(n_data_points):
    for breakpoint in range(n_breakpoints - 1):
        m.addConstr(ZF[i, breakpoint] <= Z[i, breakpoint])
        m.addConstr(ZL[i, breakpoint] <= Z[i, breakpoint])



for i in range(n_data_points - 1):
    for breakpoint in range(n_breakpoints - 2):
           m.addConstr(data[i, 0] * A[breakpoint+1] + B[breakpoint+1] - (data[i, 0] * A[breakpoint] + B[breakpoint]) == P_plus[i, breakpoint] - P_minus[i, breakpoint])
           m.addConstr(data[i+1, 0] * A[breakpoint] + B[breakpoint] - (data[i+1, 0] * A[breakpoint+1] + B[breakpoint+1]) == Q_plus[i+1, breakpoint+1] - Q_minus[i+1, breakpoint+1])

           m.addConstr(P_plus[i, breakpoint] <= M_a[i] * (1 - U[i, breakpoint]))
           m.addConstr(Q_plus[i+1, breakpoint+1] <= M_a[i] * (1 - U[i, breakpoint]))
           m.addConstr(P_minus[i, breakpoint] <= M_a[i] * (1 - V[i, breakpoint]))
           m.addConstr(Q_minus[i+1, breakpoint+1] <= M_a[i] * (1 - V[i, breakpoint]))

           m.addConstr(U[i, breakpoint] + V[i, breakpoint] == ZL[i, breakpoint])

if distance_metric == "L1":
    m.setObjective(E1, gp.GRB.MINIMIZE)

else:
    pass

m.optimize()
obj = m.getObjective()

print("Affine functions")
affine_function_str = f"""{A[0].X} *X + {B[0].X}  ({data[0,0]}  <= X <= {(B[1].X - B[0].X)/ (A[0].X - A[1].X)})\n"""
for b in range(1, n_breakpoints - 2):
   affine_function_str +=  f"{A[b].X} *X + {B[b].X}  ({(B[b].X - B[b-1].X)/(A[b-1].X - A[b].X)}  <= X <= {(B[b+1].X - B[b].X)/ (A[b].X - A[b+1].X)})\n"

affine_function_str += f"{A[n_breakpoints - 2].X} *X + {B[n_breakpoints - 2].X}  ( ({(B[n_breakpoints - 2].X - B[n_breakpoints - 3].X) / (A[n_breakpoints - 3].X - A[n_breakpoints - 2].X)})  <= X <=  {data[n_data_points - 1, 0]})\n"
print(affine_function_str)