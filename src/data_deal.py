# coding:utf-8
def transDatToList(path):
    '''
    将人工生成网络中的community.dat文件（记录每个节点属于哪几个社区）转换成
    :param path:
    :return:
    '''
    content = open(path, "r", encoding="utf-8").readlines()
    communities = {}
    for line in content:
        sent = line.strip().split("\t")
        node = sent[0]
        Communitys = sent[1].strip().split(" ")
        for community in Communitys:
            if community not in communities.keys():
                communities[community] = [node]
            else:
                communities[community].append(node)
    return list(communities.values())


def transListToDatDict(communities, nodes):
    '''
    将社区划分结果转换成{"1":[1,2],"2":[2]....}的格式
    :param communities:社区划分结果
    :param nodes:节点集
    :return:节点所属社区情况-字典
    '''
    CResultDict = {}
    for node in nodes:
        CResultDict[node] = [0] * len(communities)
    # 优化for循环之一：循环次数少的在外侧
    for c in communities:
        for node in nodes:
            if node in c:
                CResultDict[node][communities.index(c)] = communities.index(c) + 1
    return CResultDict


if __name__ == "__main__":
    transDatToList("data/benchmark/community.dat")
