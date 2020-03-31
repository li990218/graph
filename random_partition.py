import numpy as np 
import sys

def main():
    csr_partition = np.loadtxt("data/csr_partition.txt")
    dot_num = np.size(csr_partition)
    range_max = np.max(csr_partition)
    
    divide_num = int(sys.argv[1])
    part_id = 1
    a = np.zeros(dot_num)
    for i in range(dot_num):
        a[i] = part_id
        if part_id == divide_num:
            part_id = 0
        else:
            part_id += 1

    coo_connection = np.loadtxt("data/email.mtx")
    row_num = np.size(coo_connection,0)

    connect_num = 0
    for i in range(row_num):
        if a[int(coo_connection[i,0])] != a[int(coo_connection[i,1])]:
            connect_num += 1
    
    print("the not connected part is %.6f"%(connect_num/row_num))

if __name__ == "__main__":
    main()       

    
