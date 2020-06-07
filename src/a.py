# -*- coding:utf-8 -*-
'''
Created on 2017年10月28日

@summary: 利用Python实现NMI计算

@author: dreamhome
'''
import math
import numpy as np
from sklearn import metrics
def NMI(A,B):
    #样本点数
    total = len(A)
    A_ids = set(A)
    B_ids = set(B)
    #互信息计算
    MI = 0
    eps = 1.4e-45
    for idA in A_ids:
        for idB in B_ids:
            idAOccur = np.where(A==idA)
            idBOccur = np.where(B==idB)
            idABOccur = np.intersect1d(idAOccur,idBOccur)
            px = 1.0*len(idAOccur[0])/total
            py = 1.0*len(idBOccur[0])/total
            pxy = 1.0*len(idABOccur)/total
            MI = MI + pxy*math.log(pxy/(px*py)+eps,2)
    # 标准化互信息
    Hx = 0
    for idA in A_ids:
        idAOccurCount = 1.0*len(np.where(A==idA)[0])
        Hx = Hx - (idAOccurCount/total)*math.log(idAOccurCount/total+eps,2)
    Hy = 0
    for idB in B_ids:
        idBOccurCount = 1.0*len(np.where(B==idB)[0])
        Hy = Hy - (idBOccurCount/total)*math.log(idBOccurCount/total+eps,2)
    MIhat = 2.0*MI/(Hx+Hy)
    return MIhat

if __name__ == '__main__':
    A = np.array([[0, 1, 5,3,4,8,9],[1,2,3,5,4], [1, 2, 3, 4, 7, 8],[10,11,12,13,14,15]])
    B = np.array([[10,11,12,13,14,15],[0, 1, 2, 3, 4, 5, 7, 8], [0,9,6, 5]])
    print(NMI(A,B))
    print(metrics.normalized_mutual_info_score(A,B))