import scipy.io
import scipy
import pickle 
import numpy as np
from rowCluster import RowCluster
from rowCluster import Visualizer

"""
Parameters:
"""
thre = 0
mode = 'stat1'
getBitMapPre = True
getBitMapPos = True
num_groups = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
              110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
              210, 220, 230, 240, 250, 260, 270, 280, 290, 300,
              310, 320, 330, 340, 350, 360, 370, 380, 390, 400,
              410, 420, 430, 440, 450, 460, 470, 480, 490, 500]
num_groups = [50]
n_clusters = [0, 10, 20, 50, 100]
n_clusters = [50]
ratios = [50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800, 900, 1000]
Matrix_list = 'graphM.m'
Matrix_path = '../GraphM/'

with open(Matrix_list,'rb') as ff:
    Matrices = pickle.load(ff)


Results = {}
PreBitmaps = {}
PosBitmaps = {}

m = RowCluster()

if mode == 'normal':
    for idx, M_path in enumerate(Matrices):
        M = scipy.io.loadmat(Matrix_path + M_path)['Problem'][0][0]['A']
        m.setMatrix(M)
        for num_group in num_groups:
            m.setGroup(num_group)
            
            # get the original matrix's bitmap
            bitmap1, _ = m.get_bitmap(m.A, thre)
            
            # get bucket
            buckets,_ = m.getBucket(bitmap1)
            
            if getBitMapPre:
                PreBitmaps[(idx, num_group)] = bitmap1
            
            for ratio in ratios:
                m.setCluster(ratio)
                
                #get converted matrix
                labels = m.cluster(buckets)
                m.convert(buckets, labels)
                
                # get SpMV runtime
                runtime_A_1 = m.getTime(m.A)
                runtime_B_1 = m.getTime(m.B)
                runtime_A_2 = m.getTime(m.A)
                runtime_B_2 = m.getTime(m.B)
                runtime_A_3 = m.getTime(m.A)
                runtime_B_3 = m.getTime(m.B)
                runtime_A = (runtime_A_1 + runtime_A_2 + runtime_A_3) / 3
                runtime_B = (runtime_B_1 + runtime_B_2 + runtime_B_3) / 3
                
                speedup = (runtime_A - runtime_B) / runtime_A
                    
                print(idx,M_path, num_group, ratio, len(buckets) // ratio, m.A.shape, speedup)
                
                Results[(idx, num_group, ratio)] = (runtime_A, runtime_B, speedup)
                
                if getBitMapPos:
                    # get exchanged matrix's bitmap
                    bitmap2,_ = m.get_bitmap(m.B, thre)
                    
                    PosBitmaps[(idx, num_group, ratio, len(buckets) // ratio)] = bitmap2


elif mode == 'decompose': # 对角线元素分离
    for idx,M_path in enumerate(Matrices):
        
        M = scipy.io.loadmat(Matrix_path + M_path)['Problem'][0][0]['A']
        n = M.shape[0]
        c = M.shape[1]
        
        A = M.tocsr()
        
        nnz = A.data
        rowptr = A.indptr
        colidx = A.indices
        
        A1_nnz = []
        A1_rowptr = [0]
        A1_colidx = []
        
        A2_nnz = []
        A2_rowptr = [0]
        A2_colidx = []
        
        for row in range(n):
            row_start = rowptr[row]
            row_end = rowptr[row + 1]
            center_start = max(0, row - c // 512)
            center_end = min(c - 1, row + c // 512)
            count1 = 0
            count2 = 0
            for i in range(row_start, row_end):
                index = colidx[i]
                if index >= center_start and index <= center_end:
                    count1 += 1
                    A1_nnz.append(nnz[i])
                    A1_colidx.append(index)
                else:
                    count2 += 1
                    A2_nnz.append(nnz[i])
                    A2_colidx.append(index)
            A1_rowptr.append(A1_rowptr[-1] + count1)
            A2_rowptr.append(A2_rowptr[-1] + count2)
        
        
        A1 = scipy.sparse.csr_matrix((A1_nnz, A1_colidx, A1_rowptr), shape = A.shape)
        A2 = scipy.sparse.csr_matrix((A2_nnz, A2_colidx, A2_rowptr), shape = A.shape)
        
        MMs = [A, A2]
        for method in [0, 1]:
            if method == 0:
                m.setMatrix(MMs[0], False)
            else:
                m.setMatrix(MMs[1], True)
            
            for num_group in num_groups:
                
                m.setGroup(num_group)
                
                bitmap,bitmaptime = m.get_bitmap(m.A)
                
                if getBitMapPre:
                    PreBitmaps[(idx, num_group)] = bitmap1
                
                buckets,bbtime = m.getBucket(bitmap)
                
                for n_cluster in n_clusters:
                    
                    m.setCluster(n_cluster)
                    
                    labels = m.cluster(buckets)
                    
                    m.convert(buckets, labels)
                    
                    # get SpMV runtime
                    runtime_A_1 = m.getTime(m.A)
                    runtime_B_1 = m.getTime(m.B)
                    runtime_A_2 = m.getTime(m.A)
                    runtime_B_2 = m.getTime(m.B)
                    runtime_A_3 = m.getTime(m.A)
                    runtime_B_3 = m.getTime(m.B)
                    runtime_A = (runtime_A_1 + runtime_A_2 + runtime_A_3) / 3
                    runtime_B = (runtime_B_1 + runtime_B_2 + runtime_B_3) / 3
                
                    speedup = (runtime_A - runtime_B) / runtime_A
            
                    print(idx, M_path, method, num_group, n_cluster, m.A.shape, runtime_A, runtime_B, speedup)
                    
                    Results[(idx, num_group, n_cluster, method)] = (runtime_A, runtime_B, speedup)
                        
                    if getBitMapPos:
                        # get exchanged matrix's bitmap
                        bitmap2,_ = m.get_bitmap(m.B, thre)
                        
                        PosBitmaps[(idx, num_group, n_cluster)] = bitmap2

elif mode == 'stat1': #获取统计信息
    
    MM = [scipy.io.loadmat(Matrix_path + i) for i in Matrices]

    rows = []
    cols = []
    nnzs = []
    sparsitys = []
    means = []
    stds = []
    counts = []
    for idx,i in enumerate(MM):
        print(idx)
        A = i['Problem'][0][0]['A']
        A.tocsr()
        size = A.shape
        nnz = A.getnnz()
        sparsity = nnz / size[0] / size[1]
        rows.append(size[0])
        cols.append(size[1])
        nnzs.append(nnz)
        sparsitys.append(sparsity)
        rowptr = A.indptr
        nnz_dis = []
        for i in range(len(rowptr) - 1):
            nnz_dis.append(rowptr[i + 1] - rowptr[i])
        nnz_dis = np.array(nnz_dis)
        means.append(np.mean(nnz_dis))
        stds.append(np.std(nnz_dis))
        
        count = 0
        col_index = A.indices
        for row in range(size[0]):
            row_start = rowptr[row]
            row_end = rowptr[row + 1]
            center_start = max(0, row - size[1] // 500)
            center_end = min(size[1] - 1, row + size[1] // 500)
    #        print(center_start, center_end)
            for i in range(row_start, row_end):
                index = col_index[i]
                if index >= center_start and index <= center_end:
                    count += 1
        counts.append(count)
        
    stat = [rows, cols, nnzs, sparsitys, means, stds, counts]

elif mode == 'stat2':#获取统计信息
    
    MM = [scipy.io.loadmat(Matrix_path + i) for i in Matrices]

    rows = []
    bucketMean1 = [[] for i in range(len(num_groups))]
    bucketMean2 = [[] for i in range(len(num_groups))]
    bucketStd1 = [[] for i in range(len(num_groups))]
    bucketStd2 = [[] for i in range(len(num_groups))]
    for idx,i in enumerate(MM):
        
        A = i['Problem'][0][0]['A']
        A.tocsr()
        m.setMatrix(A)
        size = A.shape
        rows.append(size[0])
        for idx1, num_group in enumerate(num_groups):
            m.setGroup(num_group)
            bitmap, _ = m.get_bitmap(m.A)
            buckets,_ = m.getBucket(bitmap)
            keys = list(buckets.keys())      
            values = list(buckets.values())  
            
            rows = [len(i) for i in values]
            bucketMean1.append(np.mean(rows))
            bucketStd1.append(np.std(rows))
            
            core_keys = [i for idx,i in enumerate(keys) if len(values[idx]) > 2]
            core_rows = [len(i) for i in values if len(i) > 2]
            bucketMean2.append(np.mean(core_rows))
            bucketStd2.append(np.std(core_rows))
            
            print(idx, num_group)
            
    stat = [bucketMean1, bucketMean2, bucketStd1, bucketStd2]     
    with open('statBucket.plt', 'wb') as ff:
        pickle.dump(stat, ff)
        