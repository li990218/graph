import time
import scipy.io
import scipy
import pickle 
import numpy as np
from rowCluster import RowCluster
from rowCluster import Visualizer


def spmv(mat, vec, rp=100):
    # to warm up
    _ = mat.dot(vec)

    runtime = []
    for i in range(rp):
        start = time.time()
        _ = mat.dot(vec)
        end = time.time()
        runtime.append(end - start)
    # print(runtime)
    mean = np.mean(runtime)

    valid_runtime = []
    for i in runtime:
        if np.abs(i - mean) / mean < 0.1:
            valid_runtime.append(i)

    return np.mean(valid_runtime)


def process(mat_path, mat_name, grp_scale=10, clst_rate=10, diag_ratio=512, mode='plain', visual=True, mtime=True):
    mat = scipy.io.loadmat(mat_path + '/' + mat_name + '.mat')['Problem'][0][0]['A']
    print('========== Process in [{}] mode ==========='.format(mode))
    print(type(mat))
    if mode == 'plain':
        plain_process(mat, mat_path, mat_name, grp_scale, visual=visual, mtime=mtime)
    elif mode == 'norm':
        normal_process(mat, mat_path, mat_name, grp_scale, clst_rate, visual=visual, mtime=mtime)
    elif mode == 'diag':
        diag_process(mat, mat_path, mat_name, grp_scale, clst_rate, diag_ratio, visual=visual, mtime=mtime)


def normal_process(mat, mat_path, mat_name, grp_scale=10, clst_rate=10, visual=True, mtime=True):
    rc = RowCluster()
    rc.setMatrix(mat)
    print('[matrix]')
    print(type(rc.A), rc.A.shape)
    # print(rc.A)

    # Set the group number (bitmap width) of each row and generate the bitmap for each row
    rc.setGroup(grp_scale)
    bitmap1, _ = rc.get_bitmap(rc.A)
    np.set_printoptions(threshold=np.inf)
    print('[bitmap]')
    print(type(bitmap1), bitmap1.shape)
    # print(bitmap1)

    # Analyze the bitmaps and put the same ones into the same bucket
    buckets, _ = rc.getBucket(bitmap1)
    print('[buckets]')
    print(len(buckets))

    # Clustering buckets, the number of cluster is bucket_num/clst_rate
    rc.setCluster(clst_rate)
    labels = rc.cluster(buckets)
    print('[cluster]')
    print('{} out of {}'.format(rc.n_cluster, labels.shape))
    # print(labels)

    # Reorder the original matrix rows according to the clustering, generate a new matrix
    order = rc.convert(buckets, labels)
    print('[convert]')
    # print(len(order))
    # print(order)
    # print(type(rc.B))
    # for ind in order:
    #     print(bitmap1[ind])

    # Draw the bitmap, can set the sampling to limit the row number within 1024
    if visual:
        bitmap2, _ = rc.get_bitmap(rc.B)
        vs = Visualizer()
        plt_name = mat_name + '_norm_s{}_c{}'.format(grp_scale, rc.n_cluster)
        vs.plotSingle(bitmap2, plt_name, mat_path + '/visual')
        plt_name = mat_name + '_pre_s{}'.format(grp_scale)
        vs.plotSingle(bitmap1, plt_name, mat_path + '/visual')

    if mtime:
        t = spmv(rc.B, rc.x)
        print("[spmv]: {} in normal mode".format(t))


def plain_process(mat, mat_path, mat_name, grp_scale=10, visual=True, mtime=True):
    rc = RowCluster()
    rc.setMatrix(mat)
    print('[matrix]')
    print(type(rc.A), rc.A.shape)

    if visual:
        rc.setGroup(grp_scale)
        bitmap1, _ = rc.get_bitmap(rc.A)
        vs = Visualizer()
        plt_name = mat_name + '_pre_s{}'.format(grp_scale)
        vs.plotSingle(bitmap1, plt_name, mat_path + '/visual')

    if mtime:
        t = spmv(rc.A, rc.x)
        print("[spmv]: {} in plain mode".format(t))


def diag_process(mat, mat_path, mat_name, grp_scale=10, clst_rate=10, diag_ratio=20, visual=True, mtime=True):
    n = mat.shape[0]
    c = mat.shape[1]
    A = mat.tocsr()
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
        center_start = max(0, row - c // diag_ratio)
        center_end = min(c - 1, row + c // diag_ratio)
        # print(row, c // diag_ratio, center_start, center_end)
        count1 = 0
        count2 = 0
        for i in range(row_start, row_end):
            index = colidx[i]
            if (index >= center_start) and (index <= center_end):
                count1 += 1
                A1_nnz.append(nnz[i])
                A1_colidx.append(index)
            else:
                count2 += 1
                A2_nnz.append(nnz[i])
                A2_colidx.append(index)
        A1_rowptr.append(A1_rowptr[-1] + count1)
        A2_rowptr.append(A2_rowptr[-1] + count2)

    A1 = scipy.sparse.csr_matrix((A1_nnz, A1_colidx, A1_rowptr), shape=A.shape)
    A2 = scipy.sparse.csr_matrix((A2_nnz, A2_colidx, A2_rowptr), shape=A.shape)
    print('[Diagonal Decompose]:')
    print('Diagonal elements=', A1.nnz, 'Non-Diagonal elements=', A2.nnz)

    plain_process(A1, mat_path, mat_name+'_diag_r{}'.format(diag_ratio), grp_scale=grp_scale, visual=visual, mtime=mtime)
    normal_process(A2, mat_path, mat_name+'_non_diag_r{}'.format(diag_ratio),
                   grp_scale=grp_scale, clst_rate=clst_rate, visual=visual, mtime=mtime)


Matrix_list = 'MatrixList.m'
with open(Matrix_list, 'rb') as ff:
    Matrices = pickle.load(ff)
for ind, name in enumerate(Matrices):
    print(ind, name)

mat_path = '/home/xiatian2/Work/research/spmv/mat'
mat_name1 = 'lp_bandm'
mat_name2 = 'bp_0'
mat_name3 = '1138_bus'
mat_name4 = 'soc-sign-epinions'
mat_name5 = 'as-caida'
mat_name6 = 'Linux_call_graph'


# process(mat_path, mat_name1, grp_scale=100)
# process(mat_path, mat_name1, grp_scale=10, clst_rate=10, mode='norm')

# process(mat_path, mat_name2, grp_scale=100)
# process(mat_path, mat_name2, grp_scale=10, clst_rate=10, mode='norm')

# process(mat_path, mat_name3, grp_scale=100)
# process(mat_path, mat_name3, grp_scale=10, clst_rate=10, mode='norm')
# process(mat_path, mat_name3, grp_scale=10, clst_rate=10, diag_ratio=20, mode='diag')

# process(mat_path, mat_name4, mode='plain', mtime=False)
# process(mat_path, mat_name4, grp_scale=10, clst_rate=70, mode='norm')
# process(mat_path, mat_name4, grp_scale=20, clst_rate=700, diag_ratio=20, mode='diag')

# process(mat_path, mat_name5, mode='plain', mtime=False)
# process(mat_path, mat_name5, grp_scale=10, clst_rate=70, mode='norm', mtime=False)

process(mat_path, mat_name6, grp_scale=512, mode='plain', mtime=False)
# process(mat_path, mat_name6, grp_scale=10, clst_rate=50, mode='norm', mtime=False)
# process(mat_path, mat_name6, grp_scale=10, clst_rate=50, diag_ratio=40, mode='diag', mtime=False)