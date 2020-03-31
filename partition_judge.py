import numpy as np 

def partition_judge (csr_num, csr_addr, csr_partition):
    dot_num = len(csr_num) - 1
    row_num = csr_num[dot_num]
    part_num = np.max(csr_partition)

    part_store = np.zeros(part_num)
    connection = 0
    for i in range (dot_num):
        part_store[csr_partition[i]] += 1

        output = csr_partition[i]
        for j in range(csr_num[i]:csr_num[i+1]):
            input = csr_partition[int(csr_addr[j])]
            if output != input:
                connection += 1
    
    part_mean = np.mean(part_store)
    part2sum = 0
    for i in range(part_num):
        part2sum += (part_store[i]-part_mean)**2
    
    part2sum = part2sum/part_num

    connection = connection/csr_num[dot_num]

    return 