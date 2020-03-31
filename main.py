import numpy as np
import math as mt
from readfile import *
from mat_change import *
from distance import *
import networkx as nx
import matplotlib.pyplot as plt
from table import *

import sys
import getopt


def main():
    original_num = int(sys.argv[1])
    a = np.loadtxt('data/email.mtx')
    row_num = np.size(a,0)
    b = np.zeros([row_num,2])

    b[:,[0,1]] = a[:,[1,0]]
    np.savetxt('data/ori_graph.csv', a, delimiter = ',',fmt = '%d')
    a = np.append(a,b,axis = 0)
    a = a.astype(np.int32)
    a = a[np.lexsort(a[:,::-1].T)]
    print("get the coo matrix result:\n")
    print(a)


    csr_row_num, csr_addr, csr_partition = coo2csr(a)
    print("csr type data is \n")
    print(csr_row_num)
    print(csr_addr)
    print(csr_partition)
    np.savetxt("data/csr_row_num.txt",csr_row_num)
    ##change into csr model


    dot_num = int(a.max())
    print("there are totally %d lines %d dots\n" %(row_num,dot_num))

    csr_store = change_csr_dot(csr_row_num)
    print(csr_store)
    np.savetxt("data/csr_store.txt",csr_store)

    csr_store_sort = csr_store[np.lexsort(csr_store[:,::-1].T)]
    print("divide the data into %d pieces" %(original_num))

    print(csr_store_sort)
    np.savetxt("data/csr_store_sort.txt",csr_store_sort)

    divide_num = original_num
    partition_table = []

    print(csr_store_sort[dot_num - divide_num:dot_num-1,1])
    for i in range(divide_num):
        central_id = int(csr_store_sort[dot_num-1-i,1])
        print("the central id is %d"%(central_id))
        for j in range(csr_store_sort[dot_num-1-i,0]):
            vertex_id = int(csr_addr[csr_row_num[central_id-1]+j])
            if vertex_id in csr_store_sort[dot_num - divide_num:dot_num-1,1]:
                print("vertex_id is %d"%(vertex_id))
    
    print("after merging partitions, num is %d"%(original_num))
        
    total_include_num = 0
    ##initialize graph part
    for i in range(divide_num):
        part = part_table(csr_store_sort[dot_num-1-i,1], csr_store_sort[dot_num-1-i, 0])
        central_id = csr_store_sort[dot_num-1-i,1]
        csr_partition[central_id] = i + 1
        total_include_num += 1
        print("the number of sort vertex is %d"%(csr_store_sort[dot_num-1-i,0]))
        for j in range(csr_store_sort[dot_num-1-i,0]):
            vertex_id = int(csr_addr[csr_row_num[central_id-1]+j])
            #
            if csr_partition[vertex_id] == 0:
                part.add_vertex(vertex_id, 1)
                csr_partition[vertex_id] = i + 1
                total_include_num += 1

        partition_table.append(part)

    print("now the total num is %d"%(total_include_num))
    ##enlarging graph part

    set_loop = 0
    while total_include_num <= dot_num:
        signal_full = 0
        #print("doing loop at %d times"%(set_loop))
        set_loop += 1

        for i in range(divide_num):
            part_num = divide_num - 1 - i
            outside = len(part.vertex_id)
            part = partition_table[part_num]
            #print("the length of part vertex is %d"%(outside))
            if outside == 0:
                signal_full += 1
            else:
                for m in range(len(part.vertex_id)):
                    subvert_id = part.vertex_id[0]
                    subvert_num = csr_store[subvert_id,0]
                    #print("the number of sub part is %d"%(subvert_num))

                    for n in range(subvert_num):
                        connect_id = csr_addr[csr_row_num[subvert_id-1]+n]
                        if csr_partition[connect_id] == 0:
                            part.add_vertex(connect_id, 1)
                            total_include_num += 1
                            csr_partition[connect_id] = i + 1
                    part.drop_id()
            partition_table[part_num] = part

        if signal_full >= divide_num-1 or set_loop > dot_num:
            break
        


    print(csr_partition)
    np.savetxt("data/csr_partition.txt",csr_partition)

if __name__ == "__main__":
    main()             
        
"""
divide_num = 10
divide_num2 = 100

print("divide the data into %d pieces" %(divide_num))
print(csr_store_sort)
limit_num = csr_store_sort[dot_num - divide_num,0]
np.savetxt("data/csr_store_sort.txt",csr_store_sort)

j = 1
for i in range(divide_num):
    csr_partition[csr_store_sort[dot_num-1-i]] = j
    j += 1


for m in range(divide_num2):
    print("the small num is %d"%m)
    dot_div_id = int(csr_store_sort[dot_num-m-divide_num-1,1])
    csr_partition[dot_div_id] = j
    j = j + 1
    for i in range(divide_num):
        dot_id = int(csr_store_sort[dot_num-i-1,1])
        id_mat = csr_addr[csr_row_num[dot_id-1]:csr_row_num[dot_id]]
        id_row = expand(id_mat, dot_num, csr_row_num[dot_id] - csr_row_num[dot_id-1])

        div_id_mat = csr_addr[csr_row_num[dot_div_id-1]:csr_row_num[dot_div_id]]
        div_id_row = expand(div_id_mat, dot_num, csr_row_num[dot_div_id] - csr_row_num[dot_div_id-1])
        more_row = div_id_row&id_row
        dis = o_distance(more_row, np.zeros(dot_num))
        first_dis = o_distance(div_id_row, np.zeros(dot_num))
        next_dis = o_distance(id_row, np.zeros(dot_num))
        print("at %d part, the dis is %.6f the next_dis is %.6f\n"%(i, dis, next_dis))

        if dis >= next_dis*0.3:
            csr_partition[dot_div_id] = csr_partition[dot_id]
            print("the dot %d partition is %d"%(dot_div_id,csr_partition[dot_div_id]))
            j = j - 1
            break

    print("the dot %d partition is %d"%(dot_div_id,csr_partition[dot_div_id]))


for m in range(divide_num+divide_num2):
    dot_id = csr_store_sort[dot_num-m-1,1]
    for n in range(csr_row_num[int(dot_id)] - csr_row_num[int(dot_id)-1]):
        if csr_partition[csr_addr[csr_row_num[int(dot_id)-1] + n]] == 0:
            csr_partition[csr_addr[csr_row_num[int(dot_id)-1] + n]] = csr_partition[dot_id]

np.savetxt("data/csr_partition.txt",csr_partition)

is_part = 0
not_part = 0
for m in range(dot_num):
    if csr_partition[m] != 0:
        is_part += 1
    else:
        not_part += 1

print("the number is %.6f"%(is_part/dot_num))
G = nx.Graph()
G.add_node(range(dot_num+1))
G.add_edges_from(a)
csr_partition = csr_partition.astype(np.int32)
csr_color = csr_partition.tolist()
nx.draw_networkx(G,pos = nx.shell_layout(G), node_color = csr_color)
plt.show()


for i in range(dot_num):

for i in range(dot_num):
    if csr_partition[i] != 0:
        for j in range(csr_row_num[i+1] - csr_row_num[i]):

new_coo_graph = np.zeros([dot_num,1])

j = 1
for i in range(dot_num):
    if csr_store[i,0] > limit_num and new_coo_graph[i] == 0:
        new_coo_graph[i] = j
        j += 1

for i in range(row_num):
    if csr_store[int(a[i,0]),0] > limit_num:
        if new_coo_graph[int(a[i,1])] == 0:
            new_coo_graph[int(a[i,1])] = j
            j += 1

for n in range(dot_num):
    if new_coo_graph[n] == 0:
        new_coo_graph[n] = j
        j += 1

new_coo_graph = new_coo_graph.astype(np.int32)
np.savetxt('data/changed_id.csv',new_coo_graph, delimiter = ',',fmt = '%d')
write_raw_index('data/changed_id.csv','dot')

for m in range(row_num):
    a[m,0] = new_coo_graph[int(a[m,0])] - 1
    a[m,1] = new_coo_graph[int(a[m,1])] - 1

a = a.astype(np.int32)

np.savetxt('data/changed_coo_graph.csv', a, delimiter = ',',fmt = '%d')
write_raw_index('data/changed_coo_graph.csv','src,dst')

edge_data = gl.SFrame.read_csv("data/changed_coo_graph.csv")
vertex_data = gl.SFrame.read_csv("data/changed_id.csv")
g = SGraph(vertices=vertex_data, edges=edge_data, vid_field='dot', src_field='src', dst_field='dst')


ori_edge_data = gl.SFrame.read_csv("data/ori_graph.csv")
ori_vertex_data = gl.SFrame.read_csv("data/changed_id.csv")
ori_g = SGraph(vertices=ori_vertex_data, edges=ori_edge_data, vid_field='dot', src_field='src', dst_field='dst')
"""