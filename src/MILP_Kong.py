import itertools

import gurobipy as gp
import numpy as np
import pandas as pd


def create_and_optimize(file_path, linear_segments, objective):
    n_breakpoints = linear_segments + 1
    data = np.genfromtxt(file_path, delimiter="\t")
    n_data_points = data.shape[0]
    # Choose distance metric: feasibility, LInf, L1, L2

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

    m = gp.Model("MILP_Kong")

    Y = m.addMVar((n_data_points, n_breakpoints - 1), name="Y", lb=-float("inf"))
    c = m.addMVar(n_breakpoints - 1, name="A", lb=c_min, ub=c_max)
    d = m.addMVar(n_breakpoints - 1, name="B", lb=d_min, ub=d_max)
    E = m.addMVar(n_data_points, name="E", lb=0, ub=float("inf"))
    E1 = m.addVar(name="E1", lb=0, ub=float("inf"))

    P_plus = m.addMVar((n_data_points + 1, n_breakpoints - 1), name="P_plus", lb=0)
    P_minus = m.addMVar((n_data_points + 1, n_breakpoints - 1), name="P_minus", lb=0)

    Q_plus = m.addMVar((n_data_points + 1, n_breakpoints - 1), name="Q_plus", lb=0)
    Q_minus = m.addMVar((n_data_points + 1, n_breakpoints - 1), name="Q_minus", lb=0)

    U = m.addMVar((n_data_points + 1, n_breakpoints - 2), name="U", vtype=gp.GRB.BINARY)
    V = m.addMVar((n_data_points + 1, n_breakpoints - 2), name="V", vtype=gp.GRB.BINARY)

    Z = m.addMVar((n_data_points + 1, n_breakpoints - 1), name="Z", vtype=gp.GRB.BINARY)
    ZF = m.addMVar((n_data_points + 1, n_breakpoints - 1), name="ZF", vtype=gp.GRB.BINARY)
    ZL = m.addMVar((n_data_points + 1, n_breakpoints - 1), name="ZL", vtype=gp.GRB.BINARY)

    # Sum over bZ
    # C++ code also omits the last datapoint
    for i in range(n_data_points):
        m.addConstr(gp.quicksum(Z[i, :]) == 1)

    if objective == "LInf":
        for i in range(n_data_points):
            for breakpoint in range(n_breakpoints - 1):
                m.addConstr(
                    E1
                    >= (data[i, 0] * c[breakpoint] + d[breakpoint])
                    - data[i, 1]
                    - M_a[i] * (1 - Z[i, breakpoint])
                )
                m.addConstr(
                    E1
                    >= data[i, 1]
                    - (data[i, 0] * c[breakpoint] + d[breakpoint])
                    - M_a[i] * (1 - Z[i, breakpoint])
                )
    else:
        for i in range(n_data_points):
            for breakpoint in range(n_breakpoints - 1):
                m.addConstr(
                    E[i]
                    >= (data[i, 0] * c[breakpoint] + d[breakpoint])
                    - data[i, 1]
                    - M_a[i] * (1 - Z[i, breakpoint])
                )
                m.addConstr(
                    E[i]
                    >= data[i, 1]
                    - (data[i, 0] * c[breakpoint] + d[breakpoint])
                    - M_a[i] * (1 - Z[i, breakpoint])
                )

    for breakpoint in range(n_breakpoints - 1):
        m.addConstr(Z[0, breakpoint] == ZF[0, breakpoint])
        for i in range(1, n_data_points):
            m.addConstr(
                Z[i, breakpoint]
                == Z[i - 1, breakpoint] + ZF[i, breakpoint] - ZL[i - 1, breakpoint]
            )

    for breakpoint in range(n_breakpoints - 1):
        m.addConstr(gp.quicksum(ZL[:, breakpoint]) == 1)
        m.addConstr(gp.quicksum(ZF[:, breakpoint]) == 1)

    # partsums
    # pretty sure it should go to n_data_points and not n_data_points_1
    for i in range(n_data_points):
        for breakpoint in range(n_breakpoints - 2):
            m.addConstr(
                gp.quicksum(ZF[: i + 1, breakpoint + 1])
                <= gp.quicksum(ZF[: i + 1, breakpoint])
            )
            m.addConstr(
                gp.quicksum(ZL[: i + 1, breakpoint + 1])
                <= gp.quicksum(ZL[: i + 1, breakpoint])
            )

            # for j in range(i+1):
            # partsum_ZFa
            # partsum_ZFb
            # partsum_ZLa
            # partsum_ZLb


    for i in range(n_data_points):
        for breakpoint in range(n_breakpoints - 1):
            m.addConstr(ZF[i, breakpoint] <= Z[i, breakpoint])
            m.addConstr(ZL[i, breakpoint] <= Z[i, breakpoint])


    for i in range(n_data_points - 1):
        for breakpoint in range(n_breakpoints - 2):
            m.addConstr(
                data[i, 0] * c[breakpoint + 1]
                + d[breakpoint + 1]
                - (data[i, 0] * c[breakpoint] + d[breakpoint])
                == P_plus[i, breakpoint] - P_minus[i, breakpoint]
            )
            m.addConstr(
                data[i + 1, 0] * c[breakpoint]
                + d[breakpoint]
                - (data[i + 1, 0] * c[breakpoint + 1] + d[breakpoint + 1])
                == Q_plus[i + 1, breakpoint + 1] - Q_minus[i + 1, breakpoint + 1]
            )

            m.addConstr(P_plus[i, breakpoint] <= M_a[i] * (1 - U[i, breakpoint]))
            m.addConstr(Q_plus[i + 1, breakpoint + 1] <= M_a[i] * (1 - U[i, breakpoint]))
            m.addConstr(P_minus[i, breakpoint] <= M_a[i] * (1 - V[i, breakpoint]))
            m.addConstr(Q_minus[i + 1, breakpoint + 1] <= M_a[i] * (1 - V[i, breakpoint]))

            m.addConstr(U[i, breakpoint] + V[i, breakpoint] == ZL[i, breakpoint])

    if objective == "LInf":
        m.setObjective(E1, gp.GRB.MINIMIZE)

    if objective == "L1":
        m.setObjective(gp.quicksum(E), gp.GRB.MINIMIZE)

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
)
    return df_res



# It is assumed that the data for the independet variables ("x") is in the first column
# and the dependent variable ("y") is in the second column
if __name__ == "__main__":
    import sys
    file = "data/MpStorage50.txt"

    linear_segments = 3
    distance_metric = sys.argv[1]
    df_res = create_and_optimize(file, linear_segments, distance_metric)
    print(df_res)

