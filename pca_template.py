
# coding: utf-8
from sklearn import decomposition
import numpy as np
from numpy import *
from matplotlib import pyplot as plt
import sys

from numpy import array


def loadDataSet(fileName = 'iris_with_cluster.csv'):
    dataMat=[]
    labelMat=[]
    fr = open(fileName)
    for line in fr.readlines():
        lineArray=line.strip().split(',')
        records = []
        for attr in lineArray[:-1]:
            records.append(float(attr))
        dataMat.append(records)
        labelMat.append(int(lineArray[-1]))
    dataMat = array(dataMat)
    
    labelMat = array(labelMat)

    return dataMat,labelMat

def pca(dataMat, PC_num=2):
    '''
    Input:
        dataMat: obtained from the loadDataSet function, each row represents an observation
                 and each column represents an attribute
        PC_num:  The number of desired dimensions after applyting PCA. In this project keep it to 2.
    Output:
        lowDDataMat: the 2-d data after PCA transformation
    '''

    dataMat=dataMat-np.mean(dataMat,axis=0)
    cov=np.dot(dataMat.T,dataMat)/(dataMat.shape[0]-1)
    print("cov",cov.shape)
    eigval,eigvec=np.linalg.eig(cov)
    ind = np.argsort(eigval)[::-1]
    print("the datamat", dataMat.shape)
    print("org eigvec",eigvec.shape)
    sortedeigvec = eigvec[:,ind]
    sortedevec=sortedeigvec[:,0:2]
    # eigvec1 = eigvec[:,ind[0]]
    # eigvec2 = eigvec.T[:,ind[1]]
    eigvec1 = sortedevec[:, 0]
    eigvec2 = sortedevec[:,1]
    print("the eig",eigvec1.shape,eigvec2.shape)
    # a = np.dot( dataMat,eigvec1.T,)
    # b = np.dot( dataMat, eigvec2.T,)
    # print("ab",a.shape,b.shape)
    #lowDDataMat=np.dstack((a, b))
    lowDDataMat=np.dot(dataMat, sortedevec)
    print("lowDDataMat",lowDDataMat.shape)
    return lowDDataMat

def plot(lowDDataMat, labelMat, figname):
    '''
    Input:
        lowDDataMat: the 2-d data after PCA transformation obtained from pca function
        labelMat: the corresponding label of each observation obtained from loadData
    '''
    print(labelMat.size)
    for i in range(len(labelMat)):
        if (labelMat[i] == 0):
            print("lowdata",lowDDataMat.shape)
            one=plt.scatter( lowDDataMat[i, 1],lowDDataMat[i, 0], c="brown")

        if (labelMat[i] == 1):
            print("lowdata", lowDDataMat.shape)
            two=plt.scatter( lowDDataMat[i, 1], lowDDataMat[i, 0],c="green")
            plt.legend()
        if (labelMat[i] == 2):
            three=plt.scatter( lowDDataMat[i, 1], lowDDataMat[i, 0],c="yellow")
    plt.legend((one, two,three), ("1", "2","3"))
    plt.xlabel("second dimension")
    plt.ylabel("first dimension")
    plt.title(figname)
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) == 2:
        filename = sys.argv[1]
    else:
        filename = 'Iris_kmeans_cluster.csv'
    figname = filename
    figname = figname.replace('csv','jpg')
    dataMat, labelMat = loadDataSet(filename)
    
    lowDDataMat = pca(dataMat)
    print("low",lowDDataMat.shape)
    
    plot(lowDDataMat, labelMat, figname)
    

