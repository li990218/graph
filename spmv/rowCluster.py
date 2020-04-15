import numpy as np
import time
import cv2
import pickle
import os
import pandas as pd
import scipy.io
import scipy
from scipy import signal
from sklearn.manifold import TSNE
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 

class RowCluster:
    def __init__(self):
        pass
        
    def setMatrix(self, M, save_x = False):
        self.A = M.tocsr()
        self.nr = self.A.shape[0]
        self.nc = self.A.shape[1]
        if save_x == False:    
            self.x = np.random.normal(size=(self.A.shape[1],1))
    
    def setGroup(self, num_group):
        self.num_group = num_group
    
    def setCluster(self, ratio):
        self.ratio = ratio
        
    # Self defined SpMV computation based on CSR format
    def SpMV(self, M):
        non_zero = M.data
        col_index = M.indices
        row_ptr = M.indptr
        y = np.zeros_like(self.x)
    
        for i in range(self.nr):
            y_tmp = 0
            row_start = row_ptr[i]
            row_end = row_ptr[i + 1]
            for j in range(row_start, row_end):
                y_tmp += non_zero[j] * self.x[col_index[j]]
            y[i] = y_tmp
    
        return    

    # given Matrix M, get bitmap
    def get_bitmap(self, M, thre = 0):
        
        elements_group = self.nc // self.num_group
        bitmap = np.zeros((self.nr, self.num_group), dtype = np.int16)
        
        col_index = M.indices
        row_ptr = M.indptr
        
        if thre == 0:
            start = time.time()
            for row in range(self.nr):
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
            for row in range(self.nr):
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
        for i in range(self.nr):
            tmp = tuple(bitmap[i])
            if tmp in buckets.keys():
                buckets[tmp].append(i)
            else:
                buckets[tmp] = [i]
        
        end = time.time()
        bucket_time = end - start
        
        return buckets, bucket_time
    
    def cluster(self, buckets, method = 0):

        keys = list(buckets.keys())
        self.n_cluster = len(keys) // self.ratio
        
        if self.n_cluster <= 1:
            return
        
        mCluster = Cluster(self.n_cluster)
        
        # In this mode, cluster all buckets
        if method == 0:
           labels = mCluster.getLabel(keys)
           return labels
        # Otherwise, remove the small buckets before clustering
        else:
            values = list(buckets.values())
            # select only the buckets containing more than 2 rows 
            core_keys = [i for idx,i in enumerate(keys) if len(values[idx]) > 2]
            core_index = [idx for idx,i in enumerate(keys) if len(values[idx]) > 2]
            sparse_index = [idx for idx,i in enumerate(keys) if len(values[idx]) <= 2]
            
            core_labels = mCluster.getLabel(core_keys)
           
            full_labels = np.zeros(len(buckets))       
                   
            for i in range(len(sparse_index)):
                full_labels[sparse_index[i]] = 0
                        
            for i in range(len(core_index)):
                full_labels[core_index[i]] = core_labels[i]
            
            return full_labels
    
    def convert(self, buckets, labels):
        
        col_index = self.A.indices
        row_ptr = self.A.indptr
        non_zero = self.A.data
        
        B_non_zero = []
        B_col_index = []
        B_row_ptr = [0]
        order = []
        keys = list(buckets.keys())
        
        if self.n_cluster >= 2:
            cluster_labels = np.arange(self.n_cluster)
            for cluster in cluster_labels:
                for idx,index in enumerate(labels):
                    if index == cluster:
                        key = tuple(keys[idx])
                        rows = buckets[key]
                        order.extend(rows)
                        for row in rows:
                            row_start = row_ptr[row]
                            row_end = row_ptr[row + 1]
                            B_non_zero.extend(list(non_zero[row_start : row_end]))
                            B_col_index.extend(list(col_index[row_start : row_end]))
                            B_row_ptr.append(B_row_ptr[-1] + row_end - row_start)        
        else:  
       
            for key in keys:
                rows = buckets[key]
                for row in rows:
                    row_start = row_ptr[row]
                    row_end = row_ptr[row + 1]
                    B_non_zero.extend(list(non_zero[row_start : row_end]))
                    B_col_index.extend(list(col_index[row_start : row_end]))
                    B_row_ptr.append(B_row_ptr[-1] + row_end - row_start)
            
        self.B = scipy.sparse.csr_matrix((B_non_zero, B_col_index, B_row_ptr), shape = self.A.shape)
        
        return order
    
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
        
        for i in runtime:
            if np.abs(i - mean) / mean < 0.1:
                valid_runtime.append(i)
        
        return np.mean(valid_runtime)


class Cluster:
    def __init__(self, n_cluster, method = 0):
        self.method = method
        self.n_cluster = n_cluster
        
    def eucd(self, a, b):
        a = np.array(a)
        b = np.array(b)
        return np.linalg.norm(a - b)
    
    def GaussianBlur(self, a):    
            kernal = cv2.getGaussianKernel(5, 1, cv2.CV_64F).T
            a_blur = signal.convolve2d(a, kernal, mode="same")
            return a_blur
        
    def MyDistance(self, a,b):
        a = np.array(a)
        b = np.array(b)
        a_blur = self.GaussianBlur(a.reshape(1,-1)) + 0.00001
        b_blur = self.GaussianBlur(b.reshape(1,-1)) + 0.00001
        
#        M=(a_blur + b_blur) / 2
#        jsd = 0.5 * scipy.stats.entropy(a_blur.reshape(-1,), M.reshape(-1,)) + 90=0.5 * scipy.stats.entropy(b_blur.reshape(-1,), M.reshape(-1,))
        kl = scipy.stats.entropy(a_blur.reshape(-1,), b_blur.reshape(-1,))
        return kl
    
    def ReduceDim(self, keys, method):
        data = np.array(keys)
        if method == 0:
            tt = TSNE()
        else:
            tt = TSNE(metric = lambda a,b: self.MyDistance(a,b))
        dd = tt.fit_transform(data) 
        return dd
    
    def getDistM(self, keys):
        data = np.array(keys)
        return scipy.spatial.distance.cdist(data, data, metric = lambda a,b: self.MyDistance(a,b))
    
    def getInitLabel(self, keys):
        
        init_index = np.zeros(len(keys))
        interval = len(keys) // 7
        
        for i in range(len(keys)):
            if i <= interval:
                init_index[i] = 0
            elif i <= 2 * interval:
                init_index[i] = 1
            elif i <= 3 * interval:
                init_index[i] = 2
            elif i <= 4 * interval:
                init_index[i] = 3
            elif i <= 5 * interval:
                init_index[i] = 4
            elif i <= 6 * interval:
                init_index[i] = 5
            else:
                init_index[i] = 6
        return init_index
                
    def getLabel(self, keys, mm = None):
        data = np.array(keys)
        
        if self.method == 0:
            clf = KMeans(n_clusters = self.n_cluster)
            clf.fit(data)
        elif self.method == 1:
            clf = SpectralClustering(n_clusters = self.n_cluster, affinity='precomputed')
            clf.fit(mm)
        elif self.method == 2:
            clf = AgglomerativeClustering(n_clusters = self.n_cluster, affinity='precomputed',linkage='average')
            clf.fit(mm)
                
        return clf.labels_
    
    def plot(self, dd, index, name):
        
        dd1 = np.array([list(i) for idx,i in enumerate(dd) if index[idx] == 0])
        dd2 = np.array([list(i) for idx,i in enumerate(dd) if index[idx] == 1])
        dd3 = np.array([list(i) for idx,i in enumerate(dd) if index[idx] == 2])
        dd4 = np.array([list(i) for idx,i in enumerate(dd) if index[idx] == 3])
        dd5 = np.array([list(i) for idx,i in enumerate(dd) if index[idx] == 4])
        dd6 = np.array([list(i) for idx,i in enumerate(dd) if index[idx] == 5])
        dd7 = np.array([list(i) for idx,i in enumerate(dd) if index[idx] == 6])
        
        plt.figure()
        
        plt.plot(dd1[:,0],dd1[:,1],'r.')
        plt.plot(dd2[:,0],dd2[:,1],'g.')
        plt.plot(dd3[:,0],dd3[:,1],'b.')
        plt.plot(dd4[:,0],dd4[:,1],'y.')
        plt.plot(dd5[:,0],dd5[:,1],'c.')
        plt.plot(dd6[:,0],dd6[:,1],'m.')
        plt.plot(dd7[:,0],dd7[:,1],'k.')
        
        plt.savefig(name)
        plt.show()
    

class Visualizer:
    def __init__(self):
        pass
    
    def getSpeedup(self, infile, outfile, method = 0):
        
        """
        Transfer (index, group, cluster) -> (A_time, B_time, speedup) format dict
        to a csv file (index, group) -> (all clusters' speedup)
        """
        if type(infile) == dict:
            aa = infile
        else:  
            with open(infile, 'rb') as ff:
                aa = pickle.load(ff)
        
        keys = list(aa.keys())
    
        indexs = list(set([i[0] for i in keys]))
        groups = list(set([i[1] for i in keys]))
        clusters = list(set([i[2] for i in keys]))
        methods = [0,1]
            
        dicts = {}
        
        for index in indexs:
            for group in groups:
                key = (index, group)
                value = []
                for cluster in clusters:
                    if method == 0:
                        value.append(aa[index, group, cluster][-1])
                    else:
                        for m in methods:
                            value.append(aa[index, group, cluster, m][-1])
                        
                dicts[key] = value
        
        data = pd.DataFrame(dicts)
        data.to_csv(outfile)
        
        return dicts
    
    def getBitmap(self, Bitmaps, d):
        """
        plot bitmaps and save to current folder
        """
        keys = list(Bitmaps.keys())
        values = list(Bitmaps.values())
        for idx,key in enumerate(keys):
            self.plotBitmap(key,values[idx], d)
            
        
    def plotBitmap(self, key,bitmap, d):
        
        bitmap_sampled = self.sample(bitmap)
            
        if len(key) == 2:
            name = str(key[0]) + '_' + str(key[1]) + '_pre'
        elif len(key) == 3:
            name = str(key[0]) + '_' + str(key[1]) + '_' + str(key[2])
        elif len(key) == 4:
            name = str(key[0]) + '_' + str(key[1]) + '_' + str(key[2]) + '_' + str(key[3])
            
        print(name)    
        plt.imsave(d + '/' + name + '.png', bitmap_sampled.T, cmap = 'gray')
    
    def sample(self, bitmap):
            new_bitmap = []
            interval = bitmap.shape[0] // min(bitmap.shape[0], 1024)
            augment_factor = 512 // bitmap.shape[1]
            for i in range(bitmap.shape[0]):
                if i % interval == 0:
                    tmp = []
                    for j in bitmap[i]:
                        for k in range(augment_factor):     
                            tmp.append(j)
                    new_bitmap.append(tmp)
            return np.array(new_bitmap, dtype =np.int16)

    def plotSingle(self, bitmap, name, path, sample=True):
        if sample:
            bitmap_p = self.sample(bitmap)
        else:
            bitmap_p = bitmap
        print(path, name)
        plt.imsave(path + '/' + name + '.png', bitmap_p.T, cmap = 'gray')