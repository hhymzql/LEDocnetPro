#coding:utf-8
import networkx as nx
import math
#from src import evaluation 
from evaluation_metric import evaluation
#data: A Graph G=(V,E)
#Result: A set of overlapping community P = {C1...,Cn}
# 节点的邻居节点
def b(Graph, node):
    B = set(Graph.neighbors(node))
    return B
# 社区的邻居节点
def B(Graph, C):
    Kc = set()
    #print(C)
    for i in C:
        neighbors = set(Graph.neighbors(i))
        Kc = Kc | neighbors
    Kc = Kc - C
    return Kc

def cfc(Graph, E, B):
    
    neighbor_edge = 0
    for i in B:
        for j in B:
            if (i, j) in E:
                neighbor_edge += 1
    if len(B) - 1 == 0:
        return 0
    else:
        return 2 * neighbor_edge / (len(B)*(len(B)-1))

def BL(node_u, C, Graph):
    dist = 0  
    for node_v in C:
        dist += nx.shortest_path_length(Graph, source=node_u, target=node_v)
    dist = dist / len(C)
    b1 = b(Graph, node_u)
    D_in = len(b1 & C)
    P = len(b1) / D_in
    bl = 1 / (dist * P)
    return bl 

def IC(C,Graph):
    comp = 0
    sep = 0
    for node in C:
        for i in graph.neighbors(node):
            if i in C:
                comp += 1
            else:
                sep += 1
    comp = comp / 2
    return (comp - sep) / math.sqrt(comp + sep)

def Extension(Graph, C):
    # data: A graph G = (V, E), a community C
    # community border
    Member_degree = {}  
    Kc = B(Graph, C)
    while(Kc):
        for node in Kc:
            Member_degree[node] = BL(node, C, Graph)
        Member_degree = dict(sorted(Member_degree.items(), key=lambda d:d[1], reverse = True))
        n0 = list(Member_degree.keys())[0]
        if IC(C | {n0},Graph) > IC(C,Graph):
            C = C | {n0}
            Kc = B(Graph, C)
        else:
            Kc = {}
    return C

def DocNet(Graph, V, E):
    P = []
    imp = dict() 
    # 每个节点计算重要性,并保存到imp中
    #print(V)
    for i in V:
        B = b(Graph, i)
        imp[i] = cfc(Graph, E, B) * len(B)
    # 重要性排序重大到小，结果强制字典
    imp = dict(sorted(imp.items(), key=lambda d:d[1], reverse = True))
    while(imp):
        #print("imp",imp.keys())
        c = list(imp.keys())[0]
        C = {c} | b(Graph,c)
        C = Extension(Graph, C)
        P.append(list(C))
        for node_c in C:
            #print("node_C",node_c)
            #print(imp)
            imp.pop(node_c,0)
            #print("----------------")
            #print(imp)
    return P


if __name__ == "__main__":
    path = "data/karate.gml"  # 34个节点
    path1 = "data/dolphins.gml"  # 61个节点
    path2 = "data/polbooks.gml"  # 104个节点
    path3 = "data/football.gml"  # 114个节点
    path4 = "data/netscience.gml"  # 1588个节点
    path5 = "data/benchmark/network.dat"
    graph = nx.read_gml(path)
    #graph = nx.read_adjlist(path5)
    nodes = graph.nodes()
    node_num = graph.number_of_nodes()  # 节点数
    edge_num = graph.number_of_edges()  # 边数
    degrees = dict(graph.degree(graph))  # 各节点度数 {'1': 16,'2': 9, '3': 10...}
    edges = graph.edges()  # 边对 [('1','2'),('1','3'),....]
    Communities = DocNet(graph,nodes,edges)

    print(len(Communities))
    #print(4)
    # 社区评价公式（还未解决图邻接矩阵索引必须是整数的问题）
    print("社区评价标准：")
    print("模块度Q = ", evaluation.Modularity(Communities, edge_num, degrees, edges))
    print("改进模块度EQ = ", evaluation.ExtendQ(Communities, edge_num, degrees, edges))
    print("Qov=", evaluation.Qov(Communities, nodes, edge_num, degrees, edges))
                        
