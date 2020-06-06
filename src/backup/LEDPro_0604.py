import networkx as nx  # 导入networkx包
import time
import math
from decimal import Decimal
import numpy as np
from sklearn import metrics
from evaluation_metric import evaluation


def sim_leaderrank(graph):
    '''
    计算加了相似度的LR
    :param graph:
    :return: 降序LR_Sim
    '''
    nodes = graph.nodes()
    LR = leaderrank(graph)
    LR = dict((x, y) for x, y in LR)
    # 定义SR，LR加了相似度
    SR_Marix = dict.fromkeys(nodes, 1.0)
    sum_j = 0.0
    for id, label in enumerate(graph.nodes()):
        for node_neighbor in graph.neighbors(label):
            num = len(set(graph.neighbors(label)) & set(graph.neighbors(node_neighbor)))
            temp = (similarity(graph, label, node_neighbor) + 1) / (num + 1) * LR[node_neighbor]
            sum_j += temp
        SR_Marix[label] = LR[label] * sum_j
    SR_Marix = list(sorted(SR_Marix.items(), key=lambda e: e[1], reverse=True))
    return SR_Marix


def leaderrank(graph):
    """
    节点排序LR
    :param graph:复杂网络图Graph
    :return: 返回节点排序值
    """
    nodes = graph.nodes()
    # 节点个数
    num_nodes = graph.number_of_nodes()
    # 在网络中增加节点g并且与所有节点进行连接
    graph.add_node(0)
    for node in nodes:
        graph.add_edge(0, node)

    # LR值初始化为 1
    LR = dict.fromkeys(nodes, 1.0)
    name = dict.fromkeys(nodes, 1.0)
    for i, j in enumerate(name.items()):
        name[j[0]] = i

    LR[0] = 0.0
    # 迭代从而满足停止条件
    while True:
        tempLR = {}
        for node1 in graph.nodes():
            s = 0.0
            for node2 in graph.nodes():
                if node2 in graph.neighbors(node1):
                    # graph.degree([node2])[node2] node2的度
                    s += 1.0 / graph.degree([node2])[node2] * LR[node2]
            tempLR[node1] = s
        # 终止条件:LR值不再变化
        error = 0.0
        for n in tempLR.keys():
            error += abs(tempLR[n] - LR[n])
        if error <= 0.001:
            break
        LR = tempLR
    # 节点g的LR值平均分给其它的N个节点并且删除节点
    avg = LR[0] / num_nodes
    LR.pop(0)
    name.pop(0)
    for k in LR.keys():
        LR[k] += avg
    LR = list(sorted(LR.items(), key=lambda e: e[1], reverse=True))
    graph.remove_node(0)

    return LR


def similarity(graph, u, v):
    '''
    计算相似度
    :param u: 非核心社区 的邻接点
    :param v:
    :return: float: 相似度
    '''
    u_set = list(graph.neighbors(u))
    u_set.append(u)
    v_set = list(graph.neighbors(v))
    v_set.append(v)
    up = len(set(u_set) & set(v_set))
    down = len(set(u_set) | set(v_set))

    return up / down


def CQ(C):
    '''
    社区的评价标准
    :param C: 输入社区
    :return: CQ值
    '''
    C_in = 0.0
    C_out = 0.0
    for node in C:
        for i in graph.neighbors(node):
            if i in C:
                C_in += 1
            else:
                C_out += 1
    C_in = C_in / 2
    score = C_in / math.sqrt(pow(C_out, 2) + pow(C_in, 2))
    return score


def node_cfc(graph):
    '''
    计算节点重要性
    :param graph: 图
    :return: 重要性字典
    '''
    cfc_node = {}
    edges = list(graph.edges)
    for node in graph.nodes():
        d = graph.degree(node)
        nei_nodes = list(graph.neighbors(node))
        # count e(j,k) 邻接点中边的数量
        count = 0
        for j in nei_nodes:
            for k in nei_nodes:
                if (j, k) in edges:
                    count += 1
        # i = d * i / (d * (d - 1))
        cfc = 2 * (count / 2)
        if d == 1:
            cfc = 0
        else:
            cfc = cfc / (d - 1)
            Decimal(cfc).quantize(Decimal("0.000"))
        cfc_node[node] = cfc
    cfc_node = list(sorted(cfc_node.items(), key=lambda e: e[1], reverse=True))
    return cfc_node


def Bl(graph, Nr, U):
    '''
    计算隶属度并降序排列
    :param Nr: 初始社区
    :param U:  Nr中节点的邻节点(Nr外)
    :return:  节点对Nr的隶属度
    '''
    Bl = {}
    for i in U:
        # 计算U中u邻节点（包括在初始社区内的，U内的，外部的）
        neibor_u = set(graph.neighbors(i))
        # 计算U中u邻节点（社区内的）
        and_set = neibor_u & set(Nr)
        # 计算U中u邻节点（U内的）
        un_set = (neibor_u - set(Nr)) & set(U)
        and_sum = 0
        for k in and_set:
            and_sum += similarity(graph, i, k)
        un_sum = 0
        for m in un_set:
            un_sum += similarity(graph, i, m)
        if un_sum == 0:
            B = 0
        else:
            B = and_sum / un_sum
            Decimal(B).quantize(Decimal("0.000"))
        Bl[i] = B
    Bl = sorted(Bl.items(), key=lambda e: e[1], reverse=True)
    Bl_u = Bl.pop(0)
    return Bl_u

def LEDocnetPro(graph):
    '''
    :param graph: 输入的图
    :return:P
    '''
    P = set()
    # cfc_node_list = node_cfc(graph)
    cfc_node_list = leaderrank(graph)
    # cfc_node_list = sim_leaderrank(graph)
    print("cfc_node_list=", cfc_node_list)
    temp = 0
    while cfc_node_list != []:
        temp = temp + 1
        # 设置重要性最大的节点为初始节点
        c = cfc_node_list[0]
        # 设置重要性最大的节点的邻居节点为初始社区
        C = list(graph.neighbors(c[0]))
        C.append(c[0])
        print("初始社区：", temp, "次", C)
        Nr = C
        #获取初始社区中节点不在初始社区的邻节点
        while Nr != []:
            # 计算隶属度
            U = []
            for i in Nr:
                nbs = graph.neighbors(i)
                for k in nbs:
                    if k not in Nr:
                        U.append(k)
            # 去重
            format_U = list(set(U))
            format_U.sort(key=U.index)

            Bl_u = Bl(graph, Nr, U)
            Nr_add = Nr
            Nr_add.append(Bl_u[0])
            if CQ(Nr_add) > CQ(Nr):
                C.append(Bl_u[0])
                Nr.remove(Bl_u[0])
            else:
                Nr = []
        print(temp, "次C=", C)
        P = [x for x in P]
        P.append(C)
        print(temp, "次P=", P)
        # 将已确定社区节点在未访问节点集中删除
        cfc_node_list = {i:v for i,v in cfc_node_list}
        for i in C:
            if i in dict(cfc_node_list).keys():
                del cfc_node_list[i]
        cfc_node_list = [(x,y) for x,y in zip(cfc_node_list.keys(),cfc_node_list.values())]
        print(temp, "次collection_list=", cfc_node_list)
    return P


if __name__ == "__main__":
    time_start = time.process_time()
    path0 = "data/karate.gml"    # 34个节点
    path1 = "data/dolphins.gml"  # 61个节点
    path2 = "data/polbooks.gml"  # 104个节点
    path3 = "data/football.gml"  # 114个节点
    path4 = "data/polblogs.gml"  # 1490个节点

    path5 = "data/benchmark/network.dat"  # 1490个节点

    graph = nx.read_gml(path0)
    # graph = nx.read_adjlist(path5)
    nodes = graph.nodes()

    node_num = graph.number_of_nodes()  # 节点数
    edge_num = graph.number_of_edges()  # 边数
    degrees = dict(graph.degree(graph))  # 各节点度数 {'1': 16,'2': 9, '3': 10...}
    edges = graph.edges()  # 边对 [('1','2'),('1','3'),....]
    A = np.array(nx.adjacency_matrix(graph).todense())  # 获取图邻接矩阵

    # 执行社区发现算法
    Communities = LEDocnetPro(graph)

    # benchmark生成网络的节点所属社区列表
    bench = []
    # 本算法对比列表
    Communities_to_list = []
    # 社区评价公式（还未解决图邻接矩阵索引必须是整数的问题）
    print("社区评价标准：")
    print("模块度Q = ", evaluation.Modularity(Communities, edge_num, degrees, edges))
    print("改进模块度EQ = ", evaluation.ExtendQ(Communities, edge_num, degrees, edges))
    print("Qov=", evaluation.Qov(Communities, nodes, edge_num, degrees, edges))
    # print("NMI=",metrics.normalized_mutual_info_score(bench, Communities_to_list))

    time_end = time.process_time()
    print("time:", time_end - time_start)