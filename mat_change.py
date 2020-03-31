import numpy as np

def coo2csr(a):
    row_num = np.size(a,0)
    dot_num = int(a.max()) + 1
    row_id = np.zeros(1,int)
    row_addr = np.zeros(row_num,int)
    row_partition = np.zeros(dot_num,int)
    j = 0

    for i in range(row_num):
        if i == row_num - 1:
            j = j + 1
            row_id = np.append(row_id, j)
            row_addr[i] = a[i,1]
        elif a[i,0] == a[i+1,0]:
            j = j + 1
            row_addr[i] = a[i,1]
        else:
            j = j + 1
            row_addr[i] = a[i,1]
            row_id = np.append(row_id, j)
    
    return row_id, row_addr, row_partition
            
    
    return row_id, row_addr, row_partition

def expand(a, dot_num, range_a):
    row = np.zeros(dot_num, int)
    for i in range(range_a):
        row[a[i]] = 1
    return row

def change_csr_dot(csr_row_num):
    dot_num = np.size(csr_row_num) - 1
    csr_store = np.zeros([dot_num,3])
    for i in range(dot_num):
        csr_store[i,0] = csr_row_num[i+1] - csr_row_num[i]
        csr_store[i,1] = i + 1
    
    return csr_store.astype(np.int32)
