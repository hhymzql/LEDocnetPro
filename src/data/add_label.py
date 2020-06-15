# coding:utf-8

import networkx as nx

def gen_gml_file(gml_file, gml_file_out):
    g = nx.read_gml(gml_file,label="id")
    nodes = []
    edges = {}
    for start, end in g.edges():
        if start not in nodes:
            edges[start] = []
            edges[start].append(end)
        else:
            edges[start].append(end)

    fp_out=open(gml_file_out,'w')
    fp_out.write("graph [\n")
    for node in g.nodes():
        fp_out.write("\tnode [\n")
        fp_out.write("\t\tid "+str(node)+"\n")
        # 添加 label
        fp_out.write("\t\tlabel "+"\""+str(int(node)+1)+"\""+"\n")
        fp_out.write("\t]\n")
    
    for node1 in edges.keys():
        for node2 in edges[node1]:
            fp_out.write("\tedge [\n")
            fp_out.write("\t\tsource "+str(node1)+"\n")
            fp_out.write("\t\ttarget "+str(node2)+"\n")
            # insert other edge attributes here
            fp_out.write("\t]\n")
    fp_out.write("]\t")
 
if __name__ == "__main__":
    gen_gml_file("power.gml", "powerfull.gml")