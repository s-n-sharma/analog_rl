"""
This file produces circuits 
"""

from scipy.cluster.hierarchy import DisjointSet as djs
import Circuit as circ
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import random

def min_deg_func(dic, zero_flag = False):
    for i in dic.keys():
        if zero_flag and i == 0:
            continue
        if dic[i] < 3:
            return False
    return True

MAX_NODES = 10

counter = 0

for i in range(1, MAX_NODES):
    for _ in range(15):
        with open(f'example_circuits/graph_{counter}.txt', 'w') as f:

            djs_conn_2 = djs([j for j in range(1, i+3)]) #want to exclude connections through ground
            out_pos = set()

            EDGE_NUMBER = random.randint(int(1.5*i), int((i+3)*(i+2)*0.15)) #want graph to be pretty connected
            min_deg_2 = False
            deg_count = {k : 0 for k in range(i+3)}
            Resistor_count = 0
            Capacitor_count = 0
            op_amp_count = 0


            if _ < 8:

                OP_AMP_NUMBER = random.randint(0, int(max(0.2*EDGE_NUMBER, 1)))

                

                for op_amp_cnt in range(OP_AMP_NUMBER):
                    node_1 = random.randint(0, i+2)
                    node_2 = random.randint(0, i+2)
                    
                    while (node_1 == node_2):
                        node_2 = random.randint(0, i+2)
                    
                    node_3 = random.randint(0, i+2)

                    while (node_1 == node_3) or (node_2 == node_3) and (node_2, node_3) not in out_pos:
                        node_3 = random.randint(0, i+2)
                    
                    #djs_conn.merge(node_1, node_2)
                    #djs_conn.merge(node_2, node_3)
                    #djs_conn.merge(node_1, node_3) #not really needed lol

                    deg_count[node_1] += 1
                    deg_count[node_2] += 1
                    deg_count[node_3] += 1

                    out_pos.add((node_2, node_3))
                
                    #.subckt iop1 1 2 3
                    f.write(f"E {node_1} {node_2} {node_3}\n")
            

            
            #0 is ground, i+1 is Vin, i+2 is Vout (NVM)
            #0 is ground, 1 is Vin, 2 is Vout

            f.write(f"Vin 1 0 1\n")
            #f.write(f"Vin 0 {i+1} 1\n")

            j = 0

            while (j < EDGE_NUMBER - OP_AMP_NUMBER or  not djs_conn_2.connected(i+1, i+2) or not (len(djs_conn_2.subsets()) == 1) or not min_deg_2):
                
                component_type = random.randint(0, 1)

                if (component_type == 0):
                    Resistor_count += 1
                    component = "R"+str(Resistor_count)
                    unit = "000"
                    
                else:
                    Capacitor_count += 1
                    component = "C"+str(Capacitor_count)
                    unit = "e-9"
                
                node_1 = random.randint(1, i+2)
                node_2 = random.randint(1, i+2)

                while (node_2 == node_1):
                    node_2 = random.randint(1, i+2)
                
                if (node_1, node_2) in out_pos or (node_2, node_1) in out_pos:
                    continue

                djs_conn_2.merge(node_1, node_2)
                
                deg_count[node_1] += 1
                deg_count[node_2] += 1

                comp_strength = random.choice([0.1, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 5, 7.5, 10, 15, 20, 30, 50, 75, 100, 150, 250, 375, 500, 1000])
                f.write(f"{component} {node_1} {node_2} {str(comp_strength)+unit}\n")
                
                min_deg_2 = min_deg_func(deg_count, True)
                j += 1

            for j in range(random.randint(3, max(int(0.3*EDGE_NUMBER), 4))): #for connections to ground
                other_node = random.randint(1, i+2)

                if (component_type == 0):
                    Resistor_count += 1
                    component = "R"+str(Resistor_count)
                    unit = "000"
                    
                else:
                    Capacitor_count += 1
                    component = "C"+str(Capacitor_count)
                    unit = "e-9"
                
                
                #comp_strength = random.choice([0.1, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 5, 7.5, 10, 15, 20, 30, 50, 75, 100, 150, 250, 375, 500, 1000])
                comp_strength = 10
                f.write(f"{component} {other_node} 0 {str(comp_strength)+unit}\n")

            counter += 1