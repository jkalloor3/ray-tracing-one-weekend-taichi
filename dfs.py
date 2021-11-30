import numpy as np
import taichi as ti
import math
import networkx as nx
from time import time
from numpy.random import default_rng

# @ti.recursive
# def dfs(parallel_nodes, visited_set):
#     for node in parallel_nodes:
#         if node not in visited_set:
#             visited_set.add(node)
#             dfs(node.children)

visited_set = set()

# # switch to cpu if needed
ti.init(arch=ti.gpu)

def dfs(graph, node):
    for child in nx.descendants(graph, node):
        if child not in visited_set:
            visited_set.add(child)
            dfs(graph, child)

if __name__ == '__main__':
    # Generate random graph
    # max_degree = 10
    #
    # num_nodes = 2 ** 15
    #
    # rng = default_rng()
    #
    # # Create random dag with n nodes
    # nodes = np.ndarray((num_nodes, max_degree), dtype=np.int32)
    #
    # for i in range(num_nodes):
    #     num_edges = np.random.randint(1, max_degree, 1)
    #     size = min(num_edges, num_nodes - i - 1)
    #     edges = rng.choice(range(i+1, num_nodes), size, replace=False)
    #     add_len = int(max_degree - size)
    #     adds = [-1] * add_len
    #     edges = np.append(edges, adds)
    #     nodes[i] = edges
    #
    # np.save(f"example_{num_nodes}", nodes)

    # Perform DFS on the nodes

    adj_list = np.load("random_graphs/example_32768.npy")

    num_nodes, max_degree = adj_list.shape

    # print(adj_list.shape)

    inner_queue_1 = ti.field(dtype=ti.i32)
    visited_set = ti.field(dtype=ti.i32)
    ti.root.dense(ti.i, num_nodes).place(visited_set)
    ti.root.dynamic(ti.i, num_nodes).place(inner_queue_1)

    nodes = ti.field(dtype=ti.i32)

    ti.root.dense(ti.ij, (num_nodes, max_degree)).place(nodes)

    nodes.from_numpy(adj_list)

    num_done = 0

    @ti.kernel
    def queue_initial():
        ti.append(inner_queue_1.parent(), [], 0)


    @ti.func
    def add_children(ind: ti.i32):
        for i in range(max_degree):
            x = nodes[ind, i]
            if x > 0 and visited_set[x] != 1:
                ti.append(inner_queue_1.parent(), [], i)

    @ti.func
    def extra_function(ind: ti.i32):
        size_of_child = 0
        for i in range(max_degree):
            x = nodes[ind, i]
            size_of_child += x
            if x > 0:
                size_of_child += x
            else:
                if ind > 1:
                    for j in range(max_degree):
                        x = nodes[ind - 1, j]
                        size_of_child -= x
                else:
                    size_of_child = 7
        return size_of_child

    @ti.kernel
    def queue_big() -> ti.i32:
        add_done = 0
        for ind in range(num_done, ti.length(inner_queue_1.parent(), [])):
            visited_set[ind] = 1
            add_done += 1
            g = extra_function(ind)
            # print(g)
            add_children(ind)

        return add_done

    t = time()
    print('starting big wavefront')
    queue_initial()
    while num_done < num_nodes:
        num_done += queue_big()

    print(time() - t)

