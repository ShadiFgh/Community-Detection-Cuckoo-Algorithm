import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

max_iter = 70
num_cuckoos = 10
num_migrate = 3
num_spawning = 3
total = num_cuckoos + num_migrate + num_spawning
min_egg = 5
max_egg = 10

with open("sample dataset.txt") as file:
    # number of nodes
    n = int(file.readline())
    file.close()

# loads the given data into an array except the first line
graph = np.loadtxt("sample dataset.txt", skiprows=1, dtype=int)
# number of edges
m =len(graph)


def adjacency_matrix(graph, n):
    adj = np.zeros((n, n))
    for i in range(0, len(graph)):
        adj[graph[i][0] - 1][graph[i][1] - 1] = 1
        adj[graph[i][1] - 1][graph[i][0] - 1] = 1
    return adj

# computing degrees of nodes
def compute_degree(graph, n):
    adj = adjacency_matrix(graph, n)
    degree = np.zeros((n,1))
    for i in range(0, n):
        degree[i][0] = sum(adj[i])
    return degree

# returns neighbors of each node
def find_neighbor(graph, n):
    adj = adjacency_matrix(graph,n)
    neighbors = []
    for i in range(0, n):
        row = []
        for j in range(0, n):
            if adj[i][j] == 1:
                row.append(j+1)
        neighbors.append(row)
    return neighbors


# creating habitats
def create_habitat(graph, n, num_cuckoos):
    cuckoos = []
    for i in range(0, num_cuckoos):
        cuckoo = []
        for j in range(0, n):
            cuckoo.append(random.choice(find_neighbor(graph, n)[j]))
        cuckoos.append(cuckoo)
    return cuckoos


# if 2 nodes are in the same community, ci,j = 1, else ci,j =0
def delta(cuckoo):
    G = nx.empty_graph()
    for i in range(0, n):
        G.add_edge(i+1, cuckoo[i])
    c = np.zeros((n,n))
    for i in range(0, n):
        for j in range(0, n):
            if nx.has_path(G, i+1, j+1):
                c[i][j] = 1
    return c


def modularity_property(graph, n, cuckoos):
    adj = adjacency_matrix(graph, n)
    degree = compute_degree(graph, n)
    profit = []
    for k in range(0, len(cuckoos)):
        c = delta(cuckoos[k])
        sigma = 0
        for i in range(0, n):
            for j in range(0, n):
                    sigma = sigma + ((adj[i][j] - ((degree[i] * degree[j]) / (2 * m))) * c[i][j])
        profit.append(sigma/(2 * m))
    return np.array(profit)


