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
