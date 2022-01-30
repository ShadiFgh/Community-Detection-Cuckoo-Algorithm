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


def spawning(cuckoos):
    index = random.randint(0, num_cuckoos - 1)
    cuckoo = cuckoos[index]
    # number of eggs that the cuckoo has
    num_eggs = random.randint(min_egg, max_egg)
    for k in range(0, num_eggs):
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if cuckoo[j] in find_neighbor(graph, n)[i]:
            cuckoo[i] = cuckoo[j]
    return cuckoo


# def spawning(cuckoos, index):
#     cuckoo = cuckoos[index]
#     # number of eggs that the cuckoo has
#     num_eggs = random.randint(min_egg, max_egg)
#     for k in range(0, num_eggs):
#         i = random.randint(0, n - 1)
#         j = random.randint(0, n - 1)
#         if cuckoo[j] in find_neighbor(graph, n)[i]:
#             cuckoo[i] = cuckoo[j]
#     return cuckoo


def migration(cuckoos, profit):
    max_profit_arg = np.argmax(profit)
    target = cuckoos[max_profit_arg]
    index = random.randint(0, num_cuckoos + num_spawning - 1)
    before_migration = cuckoos[index]
    after_migration = []
    random_vector = []
    for i in range(0, n):
        random_vector.append(random.choice([0, 1]))
    for i in range(0, n):
        if random_vector[i] == 0:
            after_migration.append(target[i])
        else:
            after_migration.append(before_migration[i])
    return after_migration



cuckoos = create_habitat(graph, n, num_cuckoos)
# profit = modularity_property(graph, n, cuckoos)
max_profit = []
for iter in range(0, max_iter):
    for i in range(0, num_spawning):
        cuckoos.append(spawning(cuckoos))

    # for i in range(0, num_cuckoos):
    #     cuckoos.append(spawning(cuckoos, i))

    for i in range(0, num_migrate):
        profit1 = modularity_property(graph, n, cuckoos)
        cuckoos.append(migration(cuckoos, profit1))


    profit = modularity_property(graph, n, cuckoos)
    best_profit = max(profit)
    print(best_profit)
    max_profit.append(best_profit)
    best_arg = np.argmax(profit)
    best_cuckoo = cuckoos[best_arg]
    profitarg = np.argsort(profit.reshape((1, num_cuckoos + num_spawning + num_migrate)))
    # next generation's population
    new_pop = []
    for k in range(0, num_cuckoos):
        temp = (total - 1) - k
        new_pop.append(cuckoos[profitarg[0][temp]])
    cuckoos = new_pop


print("best cuckoo habitat:", best_cuckoo)
print("best profit:", max(max_profit))

x = np.array(range(0, max_iter))
y = np.array(max_profit).reshape((1, len(max_profit)))[0]
plt.plot(x, y, 'ro')
plt.show()