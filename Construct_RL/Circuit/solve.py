from scipy.cluster.hierarchy import DisjointSet as djs
import Circuit
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
np.seterr(all="raise")
import os


DIR = "example_circuits"
f = np.logspace(1, 6, 400)


for filename in os.listdir(DIR):
    if not os.path.isfile(os.path.join(DIR, filename)):
        continue
    file = os.path.join(DIR, filename)
    circuit = Circuit.Circuit()
    circuit.parse_netlist(file)


    failed_flag = False

    try:
        res = circuit.get_transfer_function("1", "2", f)
   
    except Exception as e:
        print(filename + str(e))
        continue

    if failed_flag:
        continue

    with open('responses/'+filename, 'a') as out:
        for i in range(len(f)):
            out.write(f'{f[i]} {res[i]}\n')




