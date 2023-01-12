import itertools
import sys

import gurobipy as gp
import numpy as np


# file = "data/MpStorage50.txt"
file = "data/eigene_data.txt"
data = np.genfromtxt(file, delimiter="\t")
n_data_points = data.shape[0]
linear_segments = 3
n_breakpoints = linear_segments + 1
# Choose distance metric: feasibility, LInf, L1, L2
# objective = "LInf"
objective = sys.argv[1]
print(objective)

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
    data[:, 1] - c_max * data[:, 0],
    data[:, 1] - c_min * data[:, 0],
]
d_max = np.max(d_c)
d_min = np.min(d_c)

M_2 = d_max - d_min - data[:, 0] * (c_min - c_max)
M_a_arr = np.c_[
    d_c - d_min,
    d_c - d_max,
]
M_a = np.max(M_a_arr, axis=1)

m = gp.Model("MILP_Rebennack")

c = m.addMVar(n_breakpoints - 1, name="c", lb=c_min, ub=c_max)
d = m.addMVar(n_breakpoints - 1, name="d", lb=d_min, ub=d_max)
gamma = m.addMVar(n_breakpoints - 2, name="gamma", vtype=gp.GRB.BINARY)
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
    ub=M_a,
    name="epsilon",
)
for i in range(n_data_points):
    m.addConstr(gp.quicksum(delta[i, :]) == 1)

for i in range(n_data_points - 1):
    m.addConstr(delta[i + 1, 0] <= delta[i, 0])
    m.addConstr(delta[i, n_breakpoints - 2] <= delta[i + 1, n_breakpoints - 2])
    for b in range(n_breakpoints - 2):
        m.addConstr(delta[i + 1, b + 1] <= delta[i, b] + delta[i, b + 1])

        m.addConstr(
            delta[i, b] + delta[i + 1, b + 1] + gamma[b] - 2 <= delta_plus[i, b]
        )
        m.addConstr(
            delta[i, b] + delta[i + 1, b + 1] + (1 - gamma[b]) - 2 <= delta_minus[i, b]
        )
        m.addConstr(
            d[b + 1] - d[b]
            >= data[i, 0] * (c[b] - c[b + 1]) - M_2[i] * (1 - delta_plus[i, b])
        )
        m.addConstr(
            d[b + 1] - d[b]
            <= data[i + 1, 0] * (c[b] - c[b + 1]) + M_2[i + 1] * (1 - delta_plus[i, b])
        )
        m.addConstr(
            d[b + 1] - d[b]
            <= data[i, 0] * (c[b] - c[b + 1]) + M_2[i] * (1 - delta_minus[i, b])
        )
        m.addConstr(
            d[b + 1] - d[b]
            >= data[i + 1, 0] * (c[b] - c[b + 1]) - M_2[i + 1] * (1 - delta_minus[i, b])
        )


if objective == "feasibility":
    m.setObjective(1)


if objective == "LInf":
    tau = m.addVar(name="tau", lb=0)
    for i in range(n_data_points):
        for b in range(n_breakpoints - 1):
            m.addConstr(
                data[i, 1] - (c[b] * data[i, 0] + d[b])
                <= tau + M_a[i] * (1 - delta[i, b])
            )
            m.addConstr(
                (c[b] * data[i, 0] + d[b]) - data[i, 1]
                <= tau + M_a[i] * (1 - delta[i, b])
            )
    m.setObjective(tau, gp.GRB.MINIMIZE)


if objective == "L1":
    for i in range(n_data_points):
        for b in range(n_breakpoints - 1):
            m.addConstr(
                data[i, 1] - (c[b] * data[i, 0] + d[b])
                <= epsilon[i] + M_a[i] * (1 - delta[i, b])
            )
            m.addConstr(
                c[b] * data[i, 0] + d[b] - data[i, 1]
                <= epsilon[i] + M_a[i] * (1 - delta[i, b])
            )

    m.setObjective(
        gp.quicksum(epsilon[i] for i in range(n_data_points)), gp.GRB.MINIMIZE
    )


if objective == "L2":
    for i in range(n_data_points):
        for b in range(n_breakpoints - 1):
            m.addConstr(
                data[i, 1] - (c[b] * data[i, 0] + d[b])
                <= epsilon[i] + M_a[i] * (1 - delta[i, b])
            )
            m.addConstr(
                c[b] * data[i, 0] + d[b] - data[i, 1]
                <= epsilon[i] + M_a[i] * (1 - delta[i, b])
            )

    m.setObjective(
        gp.quicksum(epsilon[i] ** 2 for i in range(n_data_points)), gp.GRB.MINIMIZE
    )


m.optimize()

print("Affine functions")
affine_function_str = f"""{c[0].X} *X + {d[0].X}  ({data[0,0]}  <= X <= {(d[1].X - d[0].X)/ (c[0].X - c[1].X)})\n"""
for b in range(1, n_breakpoints - 2):
    affine_function_str += f"{c[b].X} *X + {d[b].X}  ({(d[b].X - d[b-1].X)/(c[b-1].X - c[b].X)}  <= X <= {(d[b+1].X - d[b].X)/ (c[b].X - c[b+1].X)})\n"

affine_function_str += f"{c[n_breakpoints - 2].X} *X + {d[n_breakpoints - 2].X}  ( ({(d[n_breakpoints - 2].X - d[n_breakpoints - 3].X) / (c[n_breakpoints - 3].X - c[n_breakpoints - 2].X)})  <= X <=  {data[n_data_points - 1, 0]})\n"
print(affine_function_str)


def plot_data(data, func=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.scatter(x=data[:, 0], y=data[:, 1])
    ax.grid(True, axis="both")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if func is not None:
        ax.plot(data[:, 0], func(data[:, 1]), color="r")
    plt.show()



