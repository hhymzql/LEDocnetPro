import networkx as nx
from functools import reduce
import operator
from collections import Counter
import time
import data_deal as dd
import numpy as np


def Qov(C, nodes, m, degrees, edges, A):
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
    # 统计节点归属的社区数
    count_belong_communitys = dict(Counter(com))
    # 为了减少在计算节点α查询节点c是否在C中的for循环，使用类似community.dat形式存储
    node_bl_comm = dd.transListToDatDict(C, nodes)
    for c in C:
        for id1, label1 in enumerate(nodes):
            for id2, label2 in enumerate(nodes):
                # 根据条件判断循环的节点是否是否属于当前社区
                if node_bl_comm[label1] != 0:
                    A_ic = 1 / count_belong_communitys[label1]
                else:
                    A_ic = 0
                if node_bl_comm[label2] != 0:
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


if __name__ == '__main__':
    times = time.process_time()
    Communities = [
        ['9', '10', '14', '15', '16', '19', '20', '21', '23', '24', '27', '28', '29', '30', '31', '32', '33', '34',
         '25', '3', '26'],
        ['2', '3', '4', '5', '6', '7', '8', '9', '11', '12', '13', '14', '18', '20', '22', '32', '1', '29', '10', '31',
         '17']]
    path0 = "data/karate.gml"  # 34个节点
    graph = nx.read_gml(path0)
    # graph = nx.read_adjlist(path5)
    nodes = graph.nodes()

    node_num = graph.number_of_nodes()  # 节点数
    edge_num = graph.number_of_edges()  # 边数
    degrees = dict(graph.degree(graph))  # 各节点度数 {'1': 16,'2': 9, '3': 10...}
    edges = graph.edges()  # 边对 [('1','2'),('1','3'),....]
    A = np.array(nx.adjacency_matrix(graph).todense())  # 获取图邻接矩阵
    print("Qov=", Qov(Communities, nodes, edge_num, degrees, edges, A))
    timee = time.process_time()
    print("time=", timee - times)
