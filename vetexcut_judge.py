import numpy as np 

def vertex_reuse_number(coo_partition, coo_input):
    edge_num = np.size(coo_input,0)
    dot_num = np.max(coo_input)

    reuse_plot = 0
    part_sum = np.zeros(dot_num)
    for i in range(edge_num):
        if part_sum[coo_input[i,0]] == 0:
            part_sum[coo_input[i,0]] = coo_partition[i]
        elif part_sum[coo_input[i,0]] != coo_partition[i]:
            reuse_plot += 1
    
    return reuse_plot