import numpy as np
import pandas as pd
import pytest

import subprocess

import src.MILP_Rebennack as reb
import src.MILP_Kong as kong


@pytest.mark.reb
def test_output_rebennack():
    func_info_py = reb.create_and_optimize(
            "data/MpStorage50.txt",
            linear_segments=3,
            objective="L1"
    )
    subprocess.run(['./compile_pwl.sh MILP_Rebennack.cpp Output/Rebennack_out'], shell=True)
    subprocess.run(['Output/Rebennack_out'])
    func_info_cpp = pd.read_csv("res_rebennack_cpp.csv")
    assert np.allclose(
            func_info_py.astype(float).values,
            func_info_cpp.astype(float).values,
    )


@pytest.mark.kong
def test_output_kong():
    func_info_py = kong.create_and_optimize(
            "data/MpStorage50.txt",
            linear_segments=3,
            distance_metric="L1"
    )
    subprocess.run(['./compile_pwl.sh MILP_Kong.cpp Output/Kong_out'], shell=True)
    subprocess.run(['Output/Kong_out'])
    func_info_cpp = pd.read_csv("res_kong.csv")
    assert np.allclose(
            func_info_py.astype(float).values,
            func_info_cpp.astype(float).values,
    )
