import numpy as np 
csr_partition = np.loadtxt('data/csr_partition.txt')
a = np.loadtxt('data/email.mtx')
row_num = np.size(a,0)

dot_num = np.size(csr_partition)
num_connection = 0
for i in range(row_num):
    if csr_partition[int(a[i,0])] != csr_partition[int(a[i,1])]:
        num_connection += 1

num_zero = 0
for j in range(dot_num):
    if csr_partition[j] == 0:
        num_zero += 1

print("the number of 0 is %d, which is %.6f part"%(num_zero, num_zero/dot_num))

print("the total connection number is %d"%num_connection)
print("the rate is %.6f"%(num_connection/row_num))