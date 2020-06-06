# coding:utf-8
def trans(path):
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
    print(communities)
if __name__ == "__main__":
    trans("data/benchmark/community.dat")