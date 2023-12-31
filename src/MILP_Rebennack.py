import itertools
import sys

import gurobipy as gp
import numpy as np
import pandas as pd


def create_and_optimize(data, linear_segments, objective):
    n_breakpoints = linear_segments + 1
    data = np.genfromtxt(data, delimiter="\t")
    n_data_points = data.shape[0]

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


    c_data = [c[breakpoint].X for breakpoint in range(n_breakpoints - 1)]
    d_data = [d[breakpoint].X for breakpoint in range(n_breakpoints - 1)]

    lower_bound = [data[0,0]]
    uppper_bound = [(d[1].X - d[0].X)/ (c[0].X - c[1].X)]
    for breakpoint in range(1, n_breakpoints - 2):
        lower_bound.append((d[breakpoint].X - d[breakpoint-1].X)/(c[breakpoint-1].X - c[breakpoint].X))
        uppper_bound.append((d[breakpoint+1].X - d[breakpoint].X)/ (c[breakpoint].X - c[breakpoint+1].X))

    lower_bound.append((d[n_breakpoints - 2].X - d[n_breakpoints - 3].X) / (c[n_breakpoints - 3].X - c[n_breakpoints - 2].X))
    uppper_bound.append(data[n_data_points - 1, 0])

    df_res = pd.DataFrame(
        {
            "c": c_data,
            "d": d_data,
            "X_lower_bound": lower_bound,
            "X_upper_bound": uppper_bound,
        }
    ).astype(float)

    def piecewise_func(x):
        print(x)
        func = np.piecewise(
            x,
            # Use XOR since we want the value where Lower Bound <= x is True and where Upper Bound < x is False,  
            (df_res['X_lower_bound'] <= x) ^ (df_res['X_upper_bound'] < x), 
            df_res['c']*x + df_res['d'],
        )

        return func

    return df_res, piecewise_func


if __name__ == "__main__":
    data = "data/MpStorage50.txt"
    linear_segments = 3
    # Choose distance metric: feasibility, LInf, L1, L2
    objective = sys.argv[1]
    df_res, piecewise_func = create_and_optimize(data, linear_segments, objective)

    import matplotlib.pyplot as plt

    data = np.genfromtxt(data, delimiter="\t")
    piecewise_data = []
    for point in data[:, 0]:
        res = piecewise_func(point)
        piecewise_data.append(res)

    fig, ax = plt.subplots()
    ax.scatter(data[:, 0], data[:, 1], label="Raw data", color='blue')
    ax.plot(data[:, 0], piecewise_data, label='Fitted PWLF', color="orange")
    ax.legend()
    plt.show()