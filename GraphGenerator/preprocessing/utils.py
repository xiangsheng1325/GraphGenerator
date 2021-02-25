import networkx as nx
import sys


def edgelist_to_graph(path):
    try:
        graph = nx.read_edgelist(path)
        return graph
    except:
        print("Wrong path entered! Absolute path of edgelist file pxpected.")
        sys.exit(1)


def pathlist_to_graphlist(path):
    with open(path, "r") as f:
        path_list = f.readlines()
    path_list = [p.strip("\n") for p in path_list if p != "\n"]
    graph_list = [edgelist_to_graph(p) for p in path_list]
    return graph_list
