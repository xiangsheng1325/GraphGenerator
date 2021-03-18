import os
import networkx as nx


def adj_to_edgelist(sparse_adj, outputfile="data/cit_cora.txt"):
    if os.path.exists(outputfile):
        return
    with open(outputfile, "w") as w:
        w.write("# Undirected Node Graph \n")
        w.write("# "+outputfile+" (graph is undirected, each edge is saved twice)\n")
        nodes = sparse_adj.shape[0]
        edges = sparse_adj.sum()
        w.write("# Nodes: "+str(nodes)+" Edges: "+str(edges)+"\n")
        w.write("# FromNodeId	ToNodeId\n")
        G = nx.from_scipy_sparse_matrix(sparse_adj)
        for edge in list(G.edges):
            n1 = edge[0]
            n2 = edge[1]
            w.write("{}\t{}\n".format(n1, n2))
        for edge in list(G.edges):
            n1 = edge[0]
            n2 = edge[1]
            w.write("{}\t{}\n".format(n2, n1))
    return 0
