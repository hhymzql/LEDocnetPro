# coding:utf-8
def transDatToList(path):
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



if __name__ == "__main__":
    transDatToList("data/benchmark/community.dat")
