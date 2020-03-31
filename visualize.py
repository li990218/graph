import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt


def getSpeedup(infile, outfile):
    
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
        
    dicts = {}
    
    for index in indexs:
        for group in groups:
            key = (index, group)
            value = []
            for cluster in clusters:
                value.append(aa[index, group, cluster][2])
            dicts[key] = value
    
    data = pd.DataFrame(dicts)
    data.to_csv(outfile)
    
    return dicts

def getBitmap(Bitmaps):
    """
    plot bitmaps and save to current folder
    """
    keys = list(Bitmaps.keys())
    values = list(Bitmaps.values())
    for idx,key in enumerate(keys):
        plotBitmap(key,values[idx])
        
    
def plotBitmap(key,bitmap):
    
    bitmap_sampled = sample(bitmap)
    
    fig = plt.figure(figsize = (15,5))
    ax = fig.add_subplot(111)
    ax.set_xticks([])
    ax.set_yticks([])
    
    if len(key) == 2:
        name = str(key[0]) + '_' + str(key[1]) + '_pre'
    else:
        name = str(key[0]) + '_' + str(key[1]) + '_' + str(key[2])
        
    print(name)
    ax.imshow(bitmap_sampled.T, cmap='gray')
    
    plt.imsave(name + '.png',bitmap_sampled.T, cmap = 'gray')

def sample(bitmap):
        new_bitmap = []
        interval = bitmap.shape[0] // 1024
        augment_factor = 512 // bitmap.shape[1]
        for i in range(bitmap.shape[0]):
            if i % interval == 0:
                tmp = []
                for j in bitmap[i]:
                    for k in range(augment_factor):     
                        tmp.append(j)
                new_bitmap.append(tmp)
        return np.array(new_bitmap, dtype =np.int16)
        