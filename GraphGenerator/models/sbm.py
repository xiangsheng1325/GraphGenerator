import datetime
import networkx as nx
import numpy as np
import bisect, pickle
import random, argparse
import community


def sample_discrete(dist):
    # sample a discrete distribution dist with values = dist.keys() and
    # probabilities = dist.values()

    i = 0
    acc = 0
    values = {}
    probs = []
    for e in dist:
        values[i] = e
        acc += dist[e]
        probs.append(acc)
        i += 1

    rand = random.random()
    pos = bisect.bisect(probs, rand)
    return values[pos]


def get_parameters(G, method="sbm"):
    part = community.best_partition(G)
    M = {}
    for e in G.edges():
        r = part[e[0]]
        s = part[e[1]]
        el = tuple(sorted([r, s]))
        M[el] = M.get(el, 0) + 1

    g = {}
    for k, v in part.items():
        g[v] = g.get(v, []) + [k]

    k = G.degree()
    K = {}
    for c in g:
        K[c] = sum([k[i] for i in g[c]])
    if method != "sbm":
        t = dict(k)
        for e in t:
            if t[e] != 0:
                t[e] = float(t[e])/K[part[e]]
    else:
        t = part.copy()
        for c in g:
            node_list = g[c]
            prob = 1./len(node_list)
            for n in node_list:
                t[n] = prob

    return (t, M, g)


def generate_from_parameters(t, w, g):
    G = nx.Graph()
    for i in g:
        G.add_nodes_from(g[i])

    # generate num of edges
    M = w.copy()
    for c in M:
        M[c] = np.random.poisson(M[c])

    # assign edges to vertices
    edges = []
    for c in M:
        r = c[0]
        s = c[1]
        for i in range(M[c]):
            n1 = sample_discrete({j: t[j] for j in g[r]})
            n2 = sample_discrete({j: t[j] for j in g[s]})
            edges.append((n1, n2))

    G.add_edges_from(edges)
    return G


def generate(G, method, repeat=1):
    t, w, g = get_parameters(G, method)
    return [generate_from_parameters(t, w, g) for i in range(repeat)]


