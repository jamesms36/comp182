## James Sanders
## jms30
## COMP 182 Homework 4 autograder file

import random
from collections import *

# Provided Functions
def remove_edges(g, edgelist):
    """
    Remove the edges in edgelist from the graph g.

    Arguments:
    g -- undirected graph
    edgelist - list of edges in g to remove

    Returns:
    None
    """
    for edge in edgelist:
        (u, v) = tuple(edge)
        g[u].remove(v)
        g[v].remove(u)

def bfs(g, startnode):
    """
    Perform a breadth-first search on g starting at node startnode.

    Arguments:
    g -- undirected graph
    startnode - node in g to start the search from

    Returns:
    d -- distances from startnode to each node.
    n -- number of shortest paths from startnode to each node.
    """

    # Initiating dictionaries.
    d = {}
    n = {}
    q = deque()

    # Set all distances to infinity.
    for i in g.keys():
        d[i] = float("inf")

    # Setting up the initial node's properties.
    d[startnode] = 0
    n[startnode] = 1

    q.append(startnode)

    while len(q) > 0:
        j = q.popleft()

        # For every neighbor of j.
        for h in g[j]:
            if d[h] == float("inf"):
                d[h] = d[j] + 1
                n[h] = n[j]
                q.append(h)
            elif d[h] == d[j] + 1:
                n[h] = n[h] + n[j]

    return d, n

def connected_components(g):
    """
    Find all connected components in g.

    Arguments:
    g -- undirected graph

    Returns:
    A list of sets where each set is all the nodes in
    a connected component.
    """
    # Initially we have no components and all nodes remain to be
    # explored.
    components = []
    remaining = set(g.keys())

    while remaining:
        # Randomly select a remaining node and find all nodes
        # connected to that node
        node = random.choice(list(remaining))
        distances = bfs(g, node)[0]
        visited = set()
        for i in remaining:
            if distances[i] != float('inf'):
                visited.add(i)
        components.append(visited)

        # Remove all nodes in this component from the remaining
        # nodes
        remaining -= visited

    return components

class LinAl(object):
    """
    Contains code for various linear algebra data structures and operations.
    """

    @staticmethod
    def zeroes(m, n):
        """
        Returns a matrix of zeroes with dimension m x n.
        ex: la.zeroes(3,2) -> [[0,0],[0,0],[0,0]]
        """

        return [[0] * n for i in range(m)]

    @staticmethod
    def trace(matrix):
        """
        Returns the trace of a square matrix. Assumes valid input matrix.
        ex: la.trace([[1,2],[-1,0]]) -> 1.0
        """

        if len(matrix[0]) == 0:
            return 0.0

        return float(sum(matrix[i][i] for i in range(len(matrix))))

    @staticmethod
    def transpose(matrix):
        """
        Returns the transpose of a matrix. Assumes valid input matrix.
        ex: la.transpose([[1,2,3],[4,5,6]]) -> [[1,4],[2,5],[3,6]]
        """

        res = [[0] * len(matrix) for i in range(len(matrix[0]))]

        for i in range(len(matrix[0])):
            for j in range(len(matrix)):
                res[i][j] = matrix[j][i]

        return res

    @staticmethod
    def dot(a, b):
        """
        Returns the dot product of two n x 1 vectors. Assumes valid input vectors.
        ex: la.dot([1,2,3], [3,-1,4]) -> 13.0
        """

        if len(a) != len(b):
            raise Exception("Input vectors must be of same length, not %d and %d" % (len(a), len(b)))

        return float(sum([a[i] * b[i] for i in range(len(a))]))

    @staticmethod
    def multiply(A, B):
        """
        Returns the matrix product of A and B. Assumes valid input matrices.
        ex: la.multiply([[1,2],[3,4]], [[-3,4],[2,-1]]) -> [[1.0,2.0],[-1.0,8.0]]
        """

        if len(A[0]) != len(B):
            raise Exception("Matrix dimensions do not match for matrix multiplication: %d x %d and %d x %d" % (
            len(A), len(A[0]), len(B), len(B[0])))

        result = [[0] * len(B[0]) for i in range(len(A))]

        for i in range(len(A)):
            for j in range(len(B[0])):
                result[i][j] = LinAl.dot(A[i], LinAl.transpose(B)[j])

        return result

    @staticmethod
    def sum(matrix):
        """
        Returns the sum of all the elements in matrix. Assumes valid input matrix.
        ex: la.sum([[1,2],[3,4]]) -> 10.0
        """

        return float(sum([sum(row) for row in matrix]))

    @staticmethod
    def multiply_by_val(matrix, val):
        """
        Returns the result of multiply matrix by a real number val. Assumes valid
        imput matrix and that val is a real number.
        """

        new_mat = LinAl.zeroes(len(matrix), len(matrix[0]))
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                new_mat[i][j] = val * matrix[i][j]
        return new_mat


#My own functions
def compute_flow(g: dict, dist: dict, paths: dict) -> dict:
    """
    Computes the flow across all of the edges in graph g using dist and paths from BFS

    Arguments:
    g -- The graph that flow will be computed over
    dist -- distances from startnode to each node within g. Result of when bfs is run on g with startnode.
    paths -- number of shortest paths from startnode to each node within g. Result of when bfs is run on g with startnode

    Returns:
    edge_flow -- A dictionary that maps each edge in g to the flow over that edge from the given startnode
    """

    # Initiating dictionaries.
    node_flow = {}
    edge_flow = {}

    # Set all flow values to 1 if they are connected to the startnode
    for i in g.keys():
        if dist[i] != float("inf"):
            node_flow[i] = 1
        else:
            node_flow[i] = 0
        for j in g[i]:
            edge_flow[frozenset([j, i])] = 0

    # Sorts the nodes in descending order by length from start node
    distances = sorted(dist.items(), key=lambda dist: dist[1], reverse=True)

    for elm in distances:
         if elm[1] != float("inf"):
            j = elm[0]
            # For every neighbor of j.
            for h in g[j]:
                # If h is the parent node of j
                if dist[j] == dist[h] + 1:
                    edge_flow[frozenset([j, h])] = node_flow[j]*(paths[h]/paths[j])
                    node_flow[h] = node_flow[h] + node_flow[j]*(paths[h]/paths[j])

    return edge_flow

def shortest_path_edge_betweenness(g: dict) -> dict:
    """
    Computes the shortest-path-based betweenness of all edges of graph g by summing, for each edge,
     the scores that the edge receives from all runs of compute_flow

    Arguments:
    g -- The graph that shortest path edge betweenness will be computed over

    Returns:
    total_flow -- A dictionary that maps each edge in g to the sum of all flows over that edge from all
                    runs of compute_flow
    """

    total_flow = {}
    for i in g.keys():
        for j in g[i]:
            total_flow[frozenset([j, i])] = 0

    for i in g.keys():
        dist, npaths = bfs(g, i)
        flow = compute_flow(g, dist, npaths)
        for j in flow:
            total_flow[j] = total_flow[j] + flow[j]

    return total_flow

def compute_q(g: dict, c: list) -> float:
    """
    Computes q value of graph g, when divided into the specified connected components in c

    Arguments:
    g -- The graph that q will be computed for
    c -- A list of sets, where each set is all the nodes in one of the connected components of g. All elements
        in c form a partition of g

    Returns:
    q -- the fraction of the edges in the network that connect nodes of the same type (i.e., within communities edges)
            minus the expected value of the same quantity in a graph with the same community divisions but random
            connections between the nodes
    """


    components = len(c)
    d = LinAl.zeroes(components, components)
    c_map = {}
    num_edges = 0
    for i in range(components):
        for elm in c[i]:
            c_map[elm] = i
    for i in g.keys():
        for j in g[i]:
            d[c_map[i]][c_map[j]] = d[c_map[i]][c_map[j]] + 1
            if c_map[i] != c_map[j]:
                d[c_map[j]][c_map[i]] = d[c_map[j]][c_map[i]] + 1
            num_edges = num_edges + 1
    d = LinAl.multiply_by_val(d, 1/num_edges)

    q = LinAl.trace(d) - LinAl.sum(LinAl.multiply(d,d))

    return q


# Testing
# graph1 = {0:set([1,2]), 1:set([0,3]), 2:set([0,3,4]), 3:set([1,2,5]), 4:set([2,5,6]), 5:set([3,4]), 6:set([4])}
# fig3_18g = {'A': set(['B', 'C', 'D', 'E']),
#             'B': set(['A', 'C', 'F']),
#             'C': set(['A', 'B', 'F']),
#             'D': set(['A', 'G', 'H']),
#             'E': set(['A', 'H']),
#             'F': set(['B', 'C', 'I']),
#             'G': set(['D', 'I', 'J']),
#             'H': set(['D', 'E', 'J']),
#             'I': set(['F', 'G', 'K']),
#             'J': set(['G', 'H', 'K']),
#             'K': set(['I', 'J'])}
# graph3 = {1:set([2]), 2:set([1,3]), 3:set([2,4,5]), 4:set([3,5,6]), 5:set([3,4,7]), 6:set([4,7,8,9]), 7:set([5,6,8,9]),
#           8:set([6,7,9]), 9:set([6,7,8])}
# fig3_14g = {1: set([2,3]),
#             2: set([1,3]),
#             3: set([1,2,7]),
#             4: set([5,6]),
#             5: set([4,6]),
#             6: set([4,5,7]),
#             7: set([3,6,8]),
#             8: set([7,9,12]),
#             9: set([8,10,11]),
#             10: set([9,11]),
#             11: set([9,10]),
#             12: set([8,13,14]),
#             13: set([12,14]),
#             14: set([12,13])}
# fig3_15g = {1: set([2,3]),
#             2: set([1,3,4,5]),
#             3: set([1,2,4,5]),
#             4: set([2,3,5]),
#             5: set([2,3,4,6,7]),
#             6: set([5,7]),
#             7: set([5,6,8,9,10]),
#             8: set([7,9,10]),
#             9: set([7,8,10,11]),
#             10: set([7,8,9,11]),
#             11: set([9,10])}
# dist1, npaths1 = bfs(graph1, 0)
# total_flow1 = shortest_path_edge_betweenness(graph1)
# dist_18, npaths_18 = bfs(fig3_18g, 'A')
# flow_18 = compute_flow(fig3_18g, dist_18, npaths_18)
# total_flow_18 = shortest_path_edge_betweenness(fig3_18g)
# q3 = compute_q(graph3, [set([1,2]), set([3,4,5]), set([6,7,8,9])])
# q14_1 = compute_q(fig3_14g, [set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])])
# q14_2 = compute_q(fig3_14g, [set([8, 9, 10, 11, 12, 13, 14]), set([1, 2, 3, 4, 5, 6, 7])])
# q14_3 = compute_q(fig3_14g, [set([9, 10, 11]), set([4, 5, 6]), set([8]), set([7]), set([1, 2, 3]), set([12, 13, 14])])
# q14_4 = compute_q(fig3_14g, [set([7]), set([14]), set([4]), set([12]), set([2]), set([6]), set([3]), set([9]), set([5]),
#                              set([13]), set([8]), set([1]), set([10]), set([11])])
# q15_1 = compute_q(fig3_15g, [set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])])
# q15_2 = compute_q(fig3_15g, [set([1, 2, 3, 4, 5]), set([8, 9, 10, 11, 7]), set([6])])
# q15_3 = compute_q(fig3_15g, [set([8, 9, 10, 7]), set([2, 3, 4, 5]), set([1]), set([6]), set([11])])
# q15_4 = compute_q(fig3_15g, [set([2]), set([3]), set([7]), set([9]), set([6]), set([4]), set([1]), set([8]), set([10]),
#                              set([5]), set([11])])



