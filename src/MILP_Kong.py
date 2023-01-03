import gurobipy as gp
import pandas as pd


file = "data/MpStorage50.txt"
data = pd.read_csv(file, delimiter="\t", header=None)
print(data)
n_breakpoints = 5

# Distance metric
q = 1 

