import numpy as np
import time
import scipy.io
import pickle
from sklearn.cluster import KMeans
import os

class RowCluster:
    def __init__(self):
        pass
        
    def setMatrix(self, M):
        A = M['Problem'][0][0]['A']
        self.A = A.tocsr()
        self.n = self.A.shape[0]
        self.x = np.random.normal(size=(self.A.shape[1],1))
    
    def setGroup(self, num_group):
        self.num_group = num_group
    
    def setCluster(self, n_cluster):
        self.n_cluster = n_cluster
    
    # Self defined SpMV computation based on CSR format
    def SpMV(self, M):
        non_zero = M.data
        col_index = M.indices
        row_ptr = M.indptr
        y = np.zeros_like(self.x)
    
        for i in range(self.n):
            y_tmp = 0
            row_start = row_ptr[i]
            row_end = row_ptr[i + 1]
            for j in range(row_start, row_end):
                y_tmp += non_zero[j] * self.x[col_index[j]]
            y[i] = y_tmp
    
        return  
    
    # given Matrix M, get bitmap
    def get_bitmap(self, M, thre = 0):
        
        elements_group = self.n // self.num_group
        bitmap = np.zeros((self.n, self.num_group), dtype = np.int16)
        
        col_index = M.indices
        row_ptr = M.indptr
        
        if thre == 0:
            start = time.time()
            for row in range(self.n):
                row_start = row_ptr[row]
                row_end = row_ptr[row + 1]
                for i in range(row_start, row_end):
                    index = col_index[i]
                    if index // elements_group < self.num_group:
                        bitmap[row, index // elements_group] = 1 
                    else:
                        bitmap[row, self.num_group - 1] = 1
            end = time.time()
        else:
            start = time.time()
            for row in range(self.n):
                row_start = row_ptr[row]
                row_end = row_ptr[row + 1]
                for i in range(row_start, row_end):
                    index = col_index[i]
                    if index // elements_group < self.num_group:
                        bitmap[row, index // elements_group] += 1 
                    else:
                        bitmap[row, self.num_group - 1] += 1
            valid = thre * elements_group
            bitmap = np.where(bitmap > valid, 1, 0)
            end = time.time()
            
            
        bitmap_time = end - start
        
        return bitmap, bitmap_time
    
    #given bitmap, get Buckets
    def getBucket(self, bitmap):
        
        start = time.time()
        buckets = {}
        for i in range(self.n):
            tmp = tuple(bitmap[i])
            if tmp in buckets.keys():
                buckets[tmp].append(i)
            else:
                buckets[tmp] = [i]
        
        end = time.time()
        bucket_time = end - start
        
        return buckets, bucket_time
    
    # Given buckets, cluster and then rearrange matrix rows
    def convert(self, buckets):
        
        col_index = self.A.indices
        row_ptr = self.A.indptr
        non_zero = self.A.data
        
        B_non_zero = []
        B_col_index = []
        B_row_ptr = [0]
        
        keys = list(buckets.keys())
        
        cluster_time = 0
        
        # if need to cluster
        if len(keys) > self.n_cluster and self.n_cluster > 0:
            start = time.time()
            keys = np.array(keys)
            # cluster between buckets
            clf = KMeans(n_clusters=self.n_cluster)
            clf.fit(keys)
            labels = list(clf.labels_)
            centers = clf.cluster_centers_
            
            #cluster between cluster centers
            clf2 = KMeans(n_clusters = self.n_cluster // 5)
            clf2.fit(centers)
            cluster_labels = list(clf2.labels_)
            cluster_labels = [(i,idx) for idx, i in enumerate(cluster_labels)]
            cluster_labels.sort()
            cluster_labels = [i[1] for i in cluster_labels]
            
            end = time.time()
            
            cluster_time = end - start
            
            # rearrange matrix
            start = time.time()
            for cluster in cluster_labels:
                for idx,index in enumerate(labels):
                    if index == cluster:
                        key = tuple(keys[idx])
                        rows = buckets[key]
                        for row in rows:
                            row_start = row_ptr[row]
                            row_end = row_ptr[row + 1]
                            B_non_zero.extend(list(non_zero[row_start : row_end]))
                            B_col_index.extend(list(col_index[row_start : row_end]))
                            B_row_ptr.append(B_row_ptr[-1] + row_end - row_start)
            end = time.time()
            construct_time = end - start
        
        # don't need to cluster
        else:  
            start = time.time()
            for key in keys:
                rows = buckets[key]
                for row in rows:
                    row_start = row_ptr[row]
                    row_end = row_ptr[row + 1]
                    B_non_zero.extend(list(non_zero[row_start : row_end]))
                    B_col_index.extend(list(col_index[row_start : row_end]))
                    B_row_ptr.append(B_row_ptr[-1] + row_end - row_start)
                    
            end = time.time()
            construct_time = end - start
        
        
        self.B = scipy.sparse.csr_matrix((B_non_zero, B_col_index, B_row_ptr), shape = self.A.shape)
        
        return (cluster_time, construct_time)

    def getTime(self, M, method = 0):
        
        for i in range(100):
            _ = M.dot(self.x)
            
        runtime = []
        
        for i in range(1000):
            start = time.time()
            _ = M.dot(self.x)
            end = time.time()
            runtime.append(end - start)
            
        mean = np.mean(runtime)
        valid_runtime = []
        
            
        valid_runtime = []
        for i in runtime:
            if np.abs(i - mean) / mean < 0.1:
                valid_runtime.append(i)
        
        return np.mean(valid_runtime)



num_groups = [8, 16, 32, 64, 128, 256, 512]
n_clusters = [0, 10, 20, 50, 100]

Matrices = []
for aa in os.listdir("../Matrices/"):
    if aa.endswith(".mat"):
        Matrices.append(aa)
        


thre = 0
Results = {}
PreBitmaps = {}
PosBitmaps = {}

m = RowCluster()

for idx, M_path in enumerate(Matrices):
    M = scipy.io.loadmat("../Matrices/" + M_path) #45101
    m.setMatrix(M)
    for num_group in num_groups:
        factor = 0.1
        m.setGroup(num_group)
        # get the original matrix's bitmap
        bitmap1, _ = m.get_bitmap(m.A, thre)
        PreBitmaps[(idx, num_group)] = bitmap1
        for n_cluster in n_clusters:
            
            # get bucket
            buckets,_ = m.getBucket(bitmap1)
            
            #get converted matrix
            _ = m.convert(buckets)
            
            # get SpMV runtime
            runtime_A = m.getTime(m.A)
            runtime_B = m.getTime(m.B)
            speedup = (runtime_A - runtime_B) / runtime_A
                
            print('A', ' ', num_group, ' ', n_cluster, ' ', m.A.shape, ' ', runtime_A)
            print('B', ' ', num_group, ' ', n_cluster, ' ', m.B.shape, ' ', runtime_B)
            
            Results[(idx, num_group, n_cluster)] = (runtime_A, runtime_B, speedup)
            
            # get exchanged matrix's bitmap
            bitmap2,_ = m.get_bitmap(m.B, thre)
            PosBitmaps[(idx, num_group, n_cluster)] = bitmap2
                         


def savefile(dictname, filename):
    with open(filename, 'wb') as ff:
        pickle.dump(dictname, ff)

