from itertools import chain
from functools import reduce
import operator
from collections import Counter
import data_deal as dd
import numpy as np


def Modularity(C, edge_num, degrees, edges):
    '''
    计算模块度Q
    :param C: 最终社区划分结果
    :param edge_num: 边数
    :param degrees: 各节点的度数
    :param edges: 图中连边情况
    :return: Q
    '''
    Q = 0.0
    for i in range(len(C)):
        # 社区内的边数
        ei = 0
        # 遍历社区内每个顶点和后面的顶点是否有边(步长定义减小了循环次数)
        for node_i in range(len(C[i])):
            for node_j in range(node_i + 1, len(C[i])):
                if (C[i][node_i], C[i][node_j]) in edges:
                    ei += 1

        # 社区内每个点的度之和
        di = 0
        for node in range(len(C[i])):
            di += degrees[C[i][node]]

        Q = Q + (ei - (di * di) / (edge_num * 4))

    # 简化版的Q公式
    Q = Q / float(edge_num)
    return Q


def ExtendQ(C, edge_num, degrees, edges):
    '''
    #     计算改进的模块度EQ
    #     :param C: 最终社区划分结果
    #     :param edge_num: 边数
    #     :param degrees: 各节点的度数
    #     :param edges: 图中连边情况
    #     :return: EQ
    #     '''
    #     # 计算每个节点所属的社区数
    C_List = list(chain.from_iterable(C))
    nodeO = dict(Counter(C_List))

    at = 0
    kt = 0
    EQ = 0.0
    for i in range(len(C)):
        for node_i in range(len(C[i])):
            for node_j in range(len(C[i])):
                if (C[i][node_i], C[i][node_j]) in edges:
                    Aij = 1
                else:
                    Aij = 0
                Oij = float(nodeO[C[i][node_i]] * nodeO[C[i][node_j]])
                at += Aij / Oij
                kt += degrees[C[i][node_i]] * degrees[C[i][node_j]] / Oij
    EQ = (at - kt / float(2 * edge_num)) / float(2 * edge_num)
    return EQ


def Qov_adv(C, nodes, n, m, degrees, A):
    '''
    改进的计算评价标准Qov
    :param A: 图邻接矩阵
    :param C: 社区划分结果 二维数组
    :param nodes: 顶点集
    :param m: 总边数
    :param n:节点数
    :return: Qov
    '''
    Qov = 0.0
    # 二维变一维
    com = reduce(operator.add, C)
    # 统计节点归属的社区数
    count_belong_communitys = dict(Counter(com))
    for c in C:
        # 将原始计算中αic和αjc的多次循环求和提到循环外面并优化为np的数组求和
        c_list = {}
        and_list = list(set(c) & set(nodes))
        for node in nodes:
            if node in and_list:
                c_list[node] = 1 / count_belong_communitys[node]
            else:
                c_list[node] = 0
        sum = np.array(list(c_list.values())).sum()

        for id1, label1 in enumerate(nodes):
            for id2, label2 in enumerate(nodes):
                # 根据条件判断循环的节点是否是否属于当前社区
                A_ic = c_list[label1]
                A_jc = c_list[label2]
                # 计算公式F()与A[i][j]的乘积
                Fc = A[id1][id2] * (A_ic + A_jc) / 2

                # 求 B_i B_j 公式中F的累加和
                F_i_temp = (n * A_ic +  sum)/ 2
                F_j_temp = (n * A_jc + sum) / 2

                B_i = F_i_temp / n
                B_j = F_j_temp / n
                k_i = degrees[label1]
                k_j = degrees[label2]

                F = Fc - B_j * k_j * k_i * B_i / (2 * m)
                # Qov累加
                Qov += F
    # 最后结果 除2m
    return Qov / (2 * m)


def Qov(C, nodes, m, degrees, A):
    '''
    计算评价标准Qov
    :param A: 图邻接矩阵
    :param C: 社区划分结果 二维数组
    :param nodes: 顶点集
    :param m: 总边数
    :param graph:图
    :return: Qov
    '''
    Qov = 0.0
    # 二维变一维
    com = reduce(operator.add, C)
    # 统计节点归属的社区
    count_belong_communitys = dict(Counter(com))
    # 为了减少在计算节点α查询节点c是否在C中的for循环，使用类似community.dat形式存储
    node_bl_comm = dd.transListToDatDict(C, nodes)
    for c in C:
        for id1, label1 in enumerate(nodes):
            for id2, label2 in enumerate(nodes):
                # 根据条件判断循环的节点是否是否属于当前社区
                if node_bl_comm[label1][C.index(c)] != 0:
                    A_ic = 1 / count_belong_communitys[label1]
                else:
                    A_ic = 0
                if node_bl_comm[label2][C.index(c)] != 0:
                    A_jc = 1 / count_belong_communitys[label2]
                else:
                    A_jc = 0
                # 计算公式F()与A[i][j]的乘积
                Fc = A[id1][id2] * (A_ic + A_jc) / 2

                # 求 B_i B_j 公式中F的累加和
                F_i_temp, F_j_temp = 0, 0
                for id3, label3 in enumerate(nodes):
                    if label3 in c:
                        A_kc = A_lc = 1 / count_belong_communitys[label3]
                    else:
                        A_kc = A_lc = 0
                    F_i_temp += (A_ic + A_kc) / 2
                    F_j_temp += (A_jc + A_lc) / 2
                B_i = F_i_temp / len(nodes)
                B_j = F_j_temp / len(nodes)
                k_i = degrees[label1]
                k_j = degrees[label2]

                F = Fc - B_j * k_j * k_i * B_i / (2 * m)
                # Qov累加
                Qov += F
    # 最后结果 除2m
    return Qov / (2 * m)
