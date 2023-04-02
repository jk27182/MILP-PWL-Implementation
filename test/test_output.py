import numpy as np
import pandas as pd
import pytest

import subprocess

import src.MILP_Rebennack as reb
import src.MILP_Kong as kong

__OBJECTICE_MAPPING = {
        "LInf": 1, 
        "L1": 2, 
        "L2": 3, 
}

@pytest.mark.reb
@pytest.mark.parametrize("data_path, lin_seg, objective", [
    ("data/MpStorage50.txt", 3, "L1"),
    ("data/MpStorage50.txt", 3, "L2"),
    ("data/MpStorage50.txt", 3, "LInf"),
])
def test_output_rebennack(data_path, lin_seg, objective):
    func_info_py = reb.create_and_optimize(
            data_path,
            linear_segments=lin_seg,
            objective=objective,
    )
    subprocess.run([f'Output/Rebennack_out', 'data_path', 'lin_seg', f'{__OBJECTICE_MAPPING[objective]}'])
    func_info_cpp = pd.read_csv("res_rebennack_cpp.csv")
    print(func_info_py.astype(float).values)
    print(func_info_cpp.astype(float).values)
    assert np.allclose(
            func_info_py.astype(float).values,
            func_info_cpp.astype(float).values,
            atol=0.5,
            rtol=0.1,
    )


@pytest.mark.kong
@pytest.mark.parametrize("data_path, lin_seg, objective", [
    ("data/MpStorage50.txt", 3, "LInf"), #passes
    # ("data/MpStorage50.txt", 3, "L1"), #passes 
    # ("data/MpStorage50.txt", 3, "L2"), #fails, gurobi mit komischem Output

])
def test_output_kong(data_path, lin_seg, objective):
    print(data_path)
    print(lin_seg)
    print(objective)
    func_info_py = kong.create_and_optimize(
            data_path,
            linear_segments=lin_seg,
            objective=objective,
    )
    subprocess.run(['Output/Kong_out' , 'data_path', 'lin_seg', f'{__OBJECTICE_MAPPING[objective]}'])
    func_info_cpp = pd.read_csv("res_kong.csv")
    assert np.allclose(
            func_info_py.astype(float).values,
            func_info_cpp.astype(float).values,
            atol=0.5,
            rtol=0.1,
    )
