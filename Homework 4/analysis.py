## James Sanders
## jms30
## COMP 182 Homework 4 analysis file

import random
from collections import *
import itertools
import collections
import comp182
import numpy
import random
import math
import matplotlib.pyplot as plt
import pylab
import types
import time
import math
import copy

# Provided Functions
def copy_graph(g):
    """
    Return a copy of the input graph, g

    Arguments:
    g -- a graph

    Returns:
    A copy of the input graph that does not share any objects.
    """
    return copy.deepcopy(g)
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
def gn_graph_partition(g):
    """
    Partition the graph g using the Girvan-Newman method.

    Requires connected_components, shortest_path_edge_betweenness, and
    compute_q to be defined.  This function assumes/requires these
    functions to return the values specified in the homework handout.

    Arguments:
    g -- undirected graph

    Returns:
    A list of tuples where each tuple contains a Q value and a list of
    connected components.
    """
    ### Start with initial graph
    c = connected_components(g)
    q = compute_q(g, c)
    partitions = [(q, c)]

    ### Copy graph so we can partition it without destroying original
    newg = comp182.copy_graph(g)

    ### Iterate until there are no remaining edges in the graph
    while True:
        ### Compute betweenness on the current graph
        btwn = shortest_path_edge_betweenness(newg)
        if not btwn:
            ### No information was computed, we're done
            break

        ### Find all the edges with maximum betweenness and remove them
        maxbtwn = max(btwn.values())
        maxedges = [edge for edge, b in btwn.items() if b == maxbtwn]
        remove_edges(newg, maxedges)

        ### Compute the new list of connected components
        c = connected_components(newg)
        if len(c) > len(partitions[-1][1]):
            ### This is a new partitioning, compute Q and add it to
            ### the list of partitions.
            q = compute_q(g, c)
            partitions.append((q, c))

    return partitions
def read_graph(filename):
    """
    Read a graph from a file.  The file is assumed to hold a graph
    that was written via the write_graph function.

    Arguments:
    filename -- name of file that contains the graph

    Returns:
    The graph that was stored in the input file.
    """
    with open(filename) as f:
        g = eval(f.read())
    return g
def read_attributes(filename):
    """
    Code to read student attributes from the file named filename.

    The attribute file should consist of one line per student, where
    each line is composed of student, college, year, major.  These are
    all anonymized, so each field is a number.  The student number
    corresponds to the node identifier in the Rice Facebook graph.

    Arguments:
    filename -- name of file storing the attributes

    Returns:
    A dictionary with the student numbers as keys, and a dictionary of
    attributes as values.  Each attribute dictionary contains
    'college', 'year', and 'major' as keys with the obvious associated
    values.
    """
    attributes = {}
    with open(filename) as f:
        for line in f:
            # Split line into student, college, year, major
            fields = line.split()
            student = int(fields[0])
            college = int(fields[1])
            year = int(fields[2])
            major = int(fields[3])

            # Store student in the dictionary
            attributes[student] = {'college': college,
                                   'year': year,
                                   'major': major}
    return attributes


## Plotting functions
def show():
    """
    Do not use this function unless you have trouble with figures.

    It may be necessary to call this function after drawing/plotting
    all figures.  If so, it should only be called once at the end.

    Arguments:
    None

    Returns:
    None
    """
    plt.show()
def plot_dist_linear(data, title, xlabel, ylabel, filename=None):
    """
    Plot the distribution provided in data as a bar plot on a linear
    scale.

    Arguments:
    data     -- dictionary which will be plotted with the keys
                on the x axis and the values on the y axis
    title    -- title label for the plot
    xlabel   -- x axis label for the plot
    ylabel   -- y axis label for the plot
    filename -- optional name of file to which plot will be
                saved (in png format)

    Returns:
    None
    """
    _plot_dist(data, title, xlabel, ylabel, False, filename)
def plot_dist_loglog(data, title, xlabel, ylabel, filename=None):
    """
    Plot the distribution provided in data as a scatter plot on a
    loglog scale.

    Arguments:
    data     -- dictionary which will be plotted with the keys
                on the x axis and the values on the y axis
    title    -- title label for the plot
    xlabel   -- x axis label for the plot
    ylabel   -- y axis label for the plot
    filename -- optional name of file to which plot will be
                saved (in png format)

    Returns:
    None
    """
    _plot_dist(data, title, xlabel, ylabel, True, filename)
def _pow_10_round(n, up=True):
    """
    Round n to the nearest power of 10.

    Arguments:
    n  -- number to round
    up -- round up if True, down if False

    Returns:
    rounded number
    """
    if up:
        return 10 ** math.ceil(math.log(n, 10))
    else:
        return 10 ** math.floor(math.log(n, 10))
def _plot_dist(data, title, xlabel, ylabel, scatter, filename=None):
    """
    Plot the distribution provided in data.

    Arguments:
    data     -- dictionary which will be plotted with the keys
                on the x axis and the values on the y axis
    title    -- title label for the plot
    xlabel   -- x axis label for the plot
    ylabel   -- y axis label for the plot
    scatter  -- True for loglog scatter plot, False for linear bar plot
    filename -- optional name of file to which plot will be
                saved (in png format)

    Returns:
    None
    """
    ### Check that the data is a dictionary
    if not isinstance(data, dict):
        msg = "data must be a dictionary, not {0}".format(type(data).__name__)
        raise TypeError(msg)

    ### Create a new figure
    fig = pylab.figure()

    ### Plot the data
    if scatter:
        _plot_dict_scatter(data)
    else:
        _plot_dict_bar(data, 0)

    ### Label the plot
    pylab.title(title)
    pylab.xlabel(xlabel)
    pylab.ylabel(ylabel)

    ### Draw grid
    gca = pylab.gca()
    gca.yaxis.grid(True)
    gca.xaxis.grid(False)

    if scatter:
        ### Use loglog scale
        gca.set_xscale('log')
        gca.set_yscale('log')
        gca.set_xlim([_pow_10_round(min([x for x in data.keys() if x > 0]), False),
                      _pow_10_round(max(data.keys()))])
        gca.set_ylim([_pow_10_round(min([x for x in data.values() if x > 0]), False),
                      _pow_10_round(max(data.values()))])

    ### Show the plot
    fig.show()

    ### Save to file
    if filename:
        pylab.savefig(filename)
def plot_lines(data, title, xlabel, ylabel, labels=None, filename=None):
    """
    Plot a line graph with the provided data.

    Arguments:
    data     -- a list of dictionaries, each of which will be plotted
                as a line with the keys on the x axis and the values on
                the y axis.
    title    -- title label for the plot
    xlabel   -- x axis label for the plot
    ylabel   -- y axis label for the plot
    labels   -- optional list of strings that will be used for a legend
                this list must correspond to the data list
    filename -- optional name of file to which plot will be
                saved (in png format)

    Returns:
    None
    """
    ### Check that the data is a list
    if not isinstance(data, list):
        msg = "data must be a list, not {0}".format(type(data).__name__)
        raise TypeError(msg)

    ### Create a new figure
    fig = pylab.figure()

    ### Plot the data
    if labels:
        mylabels = labels[:]
        for i in range(len(data) - len(labels)):
            mylabels.append("")
        for d, l in zip(data, mylabels):
            _plot_dict_line(d, l)
        # Add legend
        pylab.legend(loc='best')
        gca = pylab.gca()
        legend = gca.get_legend()
        pylab.setp(legend.get_texts(), fontsize='medium')
    else:
        for d in data:
            _plot_dict_line(d)

    ### Set the lower y limit to 0 or the lowest number in the values
    mins = [min(l.values()) for l in data]
    ymin = min(0, min(mins))
    pylab.ylim(ymin=ymin)

    ### Label the plot
    pylab.title(title)
    pylab.xlabel(xlabel)
    pylab.ylabel(ylabel)

    ### Draw grid lines
    pylab.grid(True)

    ### Show the plot
    fig.show()

    ### Save to file
    if filename:
        pylab.savefig(filename)
def _dict2lists(data):
    """
    Convert a dictionary into a list of keys and values, sorted by
    key.

    Arguments:
    data -- dictionary

    Returns:
    A tuple of two lists: the first is the keys, the second is the values
    """
    xvals = list(data.keys())
    xvals.sort()
    yvals = []
    for x in xvals:
        yvals.append(data[x])
    return xvals, yvals
def _plot_dict_line(d, label=None):
    """
    Plot data in the dictionary d on the current plot as a line.

    Arguments:
    d     -- dictionary
    label -- optional legend label

    Returns:
    None
    """
    xvals, yvals = _dict2lists(d)
    if label:
        pylab.plot(xvals, yvals, label=label)
    else:
        pylab.plot(xvals, yvals)
def _plot_dict_bar(d, xmin=None, label=None):
    """
    Plot data in the dictionary d on the current plot as bars.

    Arguments:
    d     -- dictionary
    xmin  -- optional minimum value for x axis
    label -- optional legend label

    Returns:
    None
    """
    xvals, yvals = _dict2lists(d)
    if xmin == None:
        xmin = min(xvals) - 1
    else:
        xmin = min(xmin, min(xvals) - 1)
    if label:
        pylab.bar(xvals, yvals, align='center', label=label)
        pylab.xlim([xmin, max(xvals) + 1])
    else:
        pylab.bar(xvals, yvals, align='center')
        pylab.xlim([xmin, max(xvals) + 1])
def _plot_dict_scatter(d):
    """
    Plot data in the dictionary d on the current plot as points.

    Arguments:
    d     -- dictionary

    Returns:
    None
    """
    xvals, yvals = _dict2lists(d)
    pylab.scatter(xvals, yvals)

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

def CountFrequency(my_list):
    """
    Returns a histogram of the frequency in which every element occurs within my_list

    """

    # Creating an empty dictionary
    freq = {}
    for item in my_list:
        if (item in freq):
            freq[item] += 1
        else:
            freq[item] = 1
    return freq

# Testing
graph1 = {0:set([1,2]), 1:set([0,3]), 2:set([0,3,4]), 3:set([1,2,5]), 4:set([2,5,6]), 5:set([3,4]), 6:set([4])}
fig3_18g = {'A': set(['B', 'C', 'D', 'E']),
            'B': set(['A', 'C', 'F']),
            'C': set(['A', 'B', 'F']),
            'D': set(['A', 'G', 'H']),
            'E': set(['A', 'H']),
            'F': set(['B', 'C', 'I']),
            'G': set(['D', 'I', 'J']),
            'H': set(['D', 'E', 'J']),
            'I': set(['F', 'G', 'K']),
            'J': set(['G', 'H', 'K']),
            'K': set(['I', 'J'])}
graph3 = {1:set([2]), 2:set([1,3]), 3:set([2,4,5]), 4:set([3,5,6]), 5:set([3,4,7]), 6:set([4,7,8,9]), 7:set([5,6,8,9]),
          8:set([6,7,9]), 9:set([6,7,8])}
fig3_13g = {1: set([32, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 18, 20, 22]),
            2: set([1, 3, 4, 8, 14, 18, 20, 22, 31]),
            3: set([1, 2, 4, 33, 8, 9, 10, 14, 28, 29]),
            4: set([1, 2, 3, 8, 13, 14]),
            5: set([1, 11, 7]),
            6: set([1, 11, 17, 7]),
            7: set([1, 5, 6, 17]),
            8: set([1, 2, 3, 4]),
            9: set([1, 34, 3, 33, 31]),
            10: set([34, 3]),
            11: set([1, 5, 6]),
            12: set([1]),
            13: set([1, 4]),
            14: set([1, 2, 3, 4, 34]),
            15: set([33, 34]),
            16: set([33, 34]),
            17: set([6, 7]),
            18: set([1, 2]),
            19: set([33, 34]),
            20: set([1, 2, 34]),
            21: set([33, 34]),
            22: set([1, 2]),
            23: set([33, 34]),
            24: set([33, 26, 28, 34, 30]),
            25: set([32, 26, 28]),
            26: set([24, 25, 32]),
            27: set([34, 30]),
            28: set([24, 25, 34, 3]),
            29: set([32, 34, 3]),
            30: set([24, 33, 34, 27]),
            31: set([9, 2, 34, 33]),
            32: set([1, 34, 33, 25, 26, 29]),
            33: set([32, 34, 3, 9, 15, 16, 19, 21, 23, 24, 30, 31]),
            34: set([32, 33, 9, 10, 14, 15, 16, 19, 20, 21, 23, 24, 27, 28, 29, 30, 31])}
fig3_14g = {1: set([2,3]),
            2: set([1,3]),
            3: set([1,2,7]),
            4: set([5,6]),
            5: set([4,6]),
            6: set([4,5,7]),
            7: set([3,6,8]),
            8: set([7,9,12]),
            9: set([8,10,11]),
            10: set([9,11]),
            11: set([9,10]),
            12: set([8,13,14]),
            13: set([12,14]),
            14: set([12,13])}
fig3_15g = {1: set([2,3]),
            2: set([1,3,4,5]),
            3: set([1,2,4,5]),
            4: set([2,3,5]),
            5: set([2,3,4,6,7]),
            6: set([5,7]),
            7: set([5,6,8,9,10]),
            8: set([7,9,10]),
            9: set([7,8,10,11]),
            10: set([7,8,9,11]),
            11: set([9,10])}
c3 = [set([1,2]), set([3,4,5]), set([6,7,8,9])]
dist1, npaths1 = bfs(graph1, 0)
dist2, npaths2 = bfs(fig3_18g, 'A')
flow2 = compute_flow(fig3_18g, dist2, npaths2)
total_flow1 = shortest_path_edge_betweenness(graph1)
total_flow2 = shortest_path_edge_betweenness(fig3_18g)
q14_1 = compute_q(fig3_14g, [set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])])
q14_2 = compute_q(fig3_14g, [set([8, 9, 10, 11, 12, 13, 14]), set([1, 2, 3, 4, 5, 6, 7])])
q14_3 = compute_q(fig3_14g, [set([9, 10, 11]), set([4, 5, 6]), set([8]), set([7]), set([1, 2, 3]), set([12, 13, 14])])
q14_3 = compute_q(fig3_14g, [set([7]), set([14]), set([4]), set([12]), set([2]), set([6]), set([3]), set([9]), set([5])
    , set([13]), set([8]), set([1]), set([10]), set([11])])

times = []
times.append(time.time())





# Part 2
fig3_13g = {1: set([32, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 18, 20, 22]),
            2: set([1, 3, 4, 8, 14, 18, 20, 22, 31]),
            3: set([1, 2, 4, 33, 8, 9, 10, 14, 28, 29]),
            4: set([1, 2, 3, 8, 13, 14]),
            5: set([1, 11, 7]),
            6: set([1, 11, 17, 7]),
            7: set([1, 5, 6, 17]),
            8: set([1, 2, 3, 4]),
            9: set([1, 34, 3, 33, 31]),
            10: set([34, 3]),
            11: set([1, 5, 6]),
            12: set([1]),
            13: set([1, 4]),
            14: set([1, 2, 3, 4, 34]),
            15: set([33, 34]),
            16: set([33, 34]),
            17: set([6, 7]),
            18: set([1, 2]),
            19: set([33, 34]),
            20: set([1, 2, 34]),
            21: set([33, 34]),
            22: set([1, 2]),
            23: set([33, 34]),
            24: set([33, 26, 28, 34, 30]),
            25: set([32, 26, 28]),
            26: set([24, 25, 32]),
            27: set([34, 30]),
            28: set([24, 25, 34, 3]),
            29: set([32, 34, 3]),
            30: set([24, 33, 34, 27]),
            31: set([9, 2, 34, 33]),
            32: set([1, 34, 33, 25, 26, 29]),
            33: set([32, 34, 3, 9, 15, 16, 19, 21, 23, 24, 30, 31]),
            34: set([32, 33, 9, 10, 14, 15, 16, 19, 20, 21, 23, 24, 27, 28, 29, 30, 31])}
karate_q = gn_graph_partition(fig3_13g)
karate_q_data = {}
for i in karate_q:
    components = len(i[1])
    karate_q_data[components] = i[0]
plot_lines([karate_q_data], "Karate Club", "Number of connected components", "Q Value")


times.append(time.time())
print("Section time:")
print(len(times))
print(times[-1] - times[-2])
print("Done with Karate")



# Part 3
rice_fb = read_graph("rice-facebook.repr")
rice_fb_q = gn_graph_partition(rice_fb)
# Since gn_graph_partition(rice_fb) takes so long to run, the results from one run were copied so that it didn't have to run each time
# rice_fb_q = [(0.0, [{4099, 5, 1542, 11, 4621, 20, 532, 4632, 1048, 539, 3612, 29, 5155, 2603, 3121, 3634, 1077, 54, 2613, 5687, 2113, 578, 3143, 5193, 3577, 2639, 3668, 2141, 605, 95, 5213, 4708, 616, 1643, 4721, 5234, 5755, 3196, 638, 639, 4225, 4739, 656, 3734, 4765, 4257, 677, 2727, 4775, 5293, 3760, 1718, 3255, 4792, 185, 5308, 1728, 4289, 4805, 3782, 4817, 3282, 1235, 5338, 4827, 3292, 1245, 734, 3297, 3303, 2280, 3818, 1771, 1264, 3830, 2296, 764, 3838, 4359, 2317, 5390, 4883, 790, 5912, 5414, 4391, 813, 3374, 817, 4914, 2866, 4403, 1847, 1336, 5434, 4932, 2889, 2382, 1876, 5973, 856, 1883, 4451, 357, 1390, 4472, 4473, 891, 2429, 3458, 3463, 2444, 5519, 1938, 916, 3994, 927, 418, 4012, 4530, 953, 6079, 4544, 3525, 2502, 5575, 6096, 5584, 4050, 2002, 6105, 5082, 474, 6109, 3554, 4076, 5614, 4080, 1520, 4595, 505}]), (-0.0035565646948327867, [{4099, 5, 1542, 11, 4621, 20, 532, 4632, 1048, 539, 3612, 29, 5155, 2603, 3121, 3634, 1077, 54, 2613, 5687, 2113, 578, 3143, 5193, 3577, 2639, 3668, 605, 5213, 95, 4708, 616, 1643, 4721, 5234, 5755, 3196, 638, 639, 4225, 4739, 656, 3734, 4765, 4257, 677, 2727, 4775, 5293, 3760, 1718, 3255, 4792, 185, 5308, 1728, 4289, 4805, 3782, 4817, 3282, 1235, 5338, 4827, 3292, 1245, 734, 3297, 3303, 2280, 3818, 1771, 1264, 3830, 2296, 764, 3838, 4359, 2317, 5390, 4883, 790, 5912, 5414, 4391, 813, 3374, 817, 4914, 2866, 4403, 1847, 1336, 5434, 4932, 2889, 2382, 1876, 5973, 856, 1883, 4451, 357, 1390, 4472, 4473, 891, 2429, 3458, 3463, 2444, 5519, 1938, 916, 3994, 927, 418, 4012, 4530, 953, 6079, 4544, 3525, 2502, 5575, 6096, 5584, 4050, 2002, 6105, 5082, 474, 6109, 3554, 4076, 5614, 4080, 1520, 4595, 505}, {2141}]), (-0.00651862416698612, [{4099, 5, 1542, 11, 4621, 20, 532, 4632, 1048, 539, 3612, 29, 5155, 2603, 3121, 3634, 1077, 54, 2613, 5687, 2113, 578, 3143, 5193, 3577, 2639, 3668, 605, 5213, 95, 4708, 616, 1643, 4721, 5234, 5755, 638, 639, 4225, 4739, 656, 3734, 4765, 4257, 677, 2727, 4775, 5293, 3760, 1718, 3255, 4792, 185, 5308, 1728, 4289, 4805, 3782, 4817, 3282, 1235, 5338, 4827, 3292, 1245, 734, 3297, 3303, 2280, 3818, 1771, 1264, 3830, 2296, 764, 3838, 4359, 2317, 5390, 4883, 790, 5912, 5414, 4391, 813, 3374, 817, 4914, 2866, 4403, 1847, 1336, 5434, 4932, 2889, 2382, 1876, 5973, 856, 1883, 4451, 357, 1390, 4472, 4473, 891, 2429, 3458, 3463, 2444, 5519, 1938, 916, 3994, 927, 418, 4012, 4530, 953, 6079, 4544, 3525, 2502, 5575, 6096, 5584, 4050, 2002, 6105, 5082, 474, 6109, 3554, 4076, 5614, 4080, 1520, 4595, 505}, {2141}, {3196}]), (-0.010670391860581607, [{4099, 5, 1542, 11, 4621, 20, 532, 4632, 1048, 539, 3612, 29, 5155, 2603, 3121, 3634, 1077, 54, 2613, 5687, 2113, 578, 3143, 5193, 3577, 2639, 3668, 605, 5213, 95, 4708, 616, 1643, 4721, 5234, 5755, 638, 639, 4225, 4739, 656, 3734, 4765, 4257, 677, 2727, 4775, 5293, 3760, 1718, 3255, 4792, 185, 5308, 1728, 4289, 4805, 3782, 4817, 3282, 1235, 5338, 4827, 3292, 1245, 734, 3297, 3303, 2280, 3818, 1771, 1264, 3830, 2296, 764, 3838, 4359, 2317, 5390, 4883, 790, 5912, 5414, 4391, 813, 3374, 817, 4914, 2866, 4403, 1847, 1336, 5434, 4932, 2889, 2382, 1876, 5973, 856, 1883, 4451, 357, 1390, 4472, 4473, 891, 2429, 3458, 3463, 2444, 5519, 916, 3994, 927, 418, 4012, 4530, 953, 6079, 4544, 3525, 2502, 5575, 6096, 5584, 4050, 2002, 6105, 5082, 474, 6109, 3554, 4076, 5614, 4080, 1520, 4595, 505}, {3196}, {1938}, {2141}]), (-0.014226956555414616, [{4099, 5, 1542, 11, 4621, 20, 532, 4632, 1048, 539, 3612, 29, 5155, 2603, 3121, 3634, 1077, 54, 2613, 5687, 2113, 578, 3143, 5193, 3577, 2639, 3668, 605, 5213, 95, 4708, 616, 1643, 4721, 5234, 5755, 638, 639, 4739, 656, 3734, 4765, 4257, 677, 2727, 4775, 5293, 3760, 1718, 3255, 4792, 185, 5308, 1728, 4289, 4805, 3782, 4817, 3282, 1235, 5338, 4827, 3292, 1245, 734, 3297, 3303, 2280, 3818, 1771, 1264, 3830, 2296, 764, 3838, 4359, 2317, 5390, 4883, 790, 5912, 5414, 4391, 813, 3374, 817, 4914, 2866, 4403, 1847, 1336, 5434, 4932, 2889, 2382, 1876, 5973, 856, 1883, 4451, 357, 1390, 4472, 4473, 891, 2429, 3458, 3463, 2444, 5519, 916, 3994, 927, 418, 4012, 4530, 953, 6079, 4544, 3525, 2502, 5575, 6096, 5584, 4050, 2002, 6105, 5082, 474, 6109, 3554, 4076, 5614, 4080, 1520, 4595, 505}, {3196}, {2141}, {4225}, {1938}]), (-0.01837872424900988, [{4099, 5, 1542, 11, 4621, 20, 532, 4632, 1048, 539, 3612, 29, 5155, 2603, 3121, 3634, 1077, 54, 2613, 5687, 2113, 578, 3143, 5193, 3577, 2639, 3668, 605, 5213, 95, 4708, 616, 1643, 4721, 5234, 5755, 638, 639, 4739, 656, 3734, 4765, 4257, 677, 2727, 4775, 5293, 3760, 1718, 3255, 4792, 185, 5308, 1728, 4289, 4805, 3782, 4817, 3282, 1235, 5338, 4827, 3292, 1245, 734, 3297, 3303, 2280, 3818, 1771, 1264, 3830, 2296, 764, 3838, 4359, 2317, 5390, 4883, 790, 5912, 5414, 4391, 813, 3374, 817, 4914, 2866, 4403, 1847, 1336, 5434, 4932, 2889, 2382, 1876, 5973, 856, 1883, 4451, 357, 1390, 4472, 4473, 891, 2429, 3458, 3463, 2444, 5519, 916, 3994, 927, 418, 4012, 4530, 953, 6079, 4544, 3525, 2502, 5575, 6096, 5584, 4050, 2002, 6105, 5082, 474, 6109, 3554, 4076, 5614, 4080, 4595, 505}, {1938}, {1520}, {3196}, {4225}, {2141}]), (-0.020758838467977103, [{4099, 5, 1542, 11, 4621, 20, 532, 4632, 1048, 539, 3612, 29, 5155, 2603, 3121, 3634, 1077, 54, 2613, 5687, 2113, 578, 3143, 5193, 3577, 2639, 3668, 605, 5213, 95, 4708, 616, 1643, 4721, 5234, 5755, 638, 639, 4739, 656, 3734, 4765, 4257, 677, 2727, 4775, 5293, 3760, 1718, 3255, 4792, 185, 5308, 1728, 4289, 4805, 3782, 4817, 3282, 1235, 5338, 4827, 3292, 1245, 734, 3297, 3303, 2280, 3818, 1771, 1264, 3830, 2296, 764, 3838, 4359, 2317, 5390, 790, 5912, 5414, 4391, 813, 3374, 817, 4914, 2866, 4403, 1847, 1336, 5434, 4932, 2889, 2382, 1876, 5973, 856, 1883, 4451, 357, 1390, 4472, 4473, 891, 2429, 3458, 3463, 2444, 5519, 916, 3994, 927, 418, 4012, 4530, 953, 6079, 4544, 3525, 2502, 5575, 6096, 5584, 4050, 2002, 6105, 5082, 474, 6109, 3554, 4076, 5614, 4080, 4595, 505}, {4225}, {3196}, {1520}, {1938}, {4883}, {2141}]), (-0.02492944611581205, [{4099, 5, 1542, 11, 4621, 20, 532, 4632, 1048, 539, 3612, 29, 5155, 2603, 3121, 3634, 1077, 54, 2613, 5687, 2113, 578, 3143, 5193, 3577, 2639, 3668, 605, 5213, 95, 4708, 616, 1643, 4721, 5234, 5755, 638, 639, 4739, 656, 3734, 4765, 4257, 677, 2727, 4775, 5293, 3760, 1718, 3255, 4792, 185, 5308, 1728, 4289, 4805, 3782, 4817, 3282, 1235, 5338, 4827, 3292, 1245, 734, 3297, 3303, 2280, 3818, 1771, 1264, 3830, 2296, 764, 3838, 4359, 2317, 5390, 790, 5912, 5414, 4391, 813, 3374, 817, 4914, 2866, 4403, 1847, 1336, 5434, 4932, 2889, 2382, 1876, 5973, 856, 1883, 4451, 357, 1390, 4472, 4473, 891, 2429, 3458, 3463, 2444, 5519, 916, 3994, 927, 418, 4012, 4530, 953, 6079, 4544, 3525, 2502, 5575, 6096, 5584, 4050, 2002, 6105, 5082, 474, 6109, 3554, 4076, 5614, 4080, 505}, {2141}, {4225}, {4883}, {1520}, {4595}, {1938}, {3196}]), (-0.029677114584252928, [{4099, 5, 1542, 11, 4621, 20, 532, 4632, 1048, 539, 3612, 29, 5155, 2603, 3121, 3634, 1077, 54, 2613, 5687, 2113, 578, 3143, 5193, 3577, 2639, 3668, 605, 5213, 95, 4708, 616, 1643, 4721, 5234, 5755, 638, 639, 4739, 656, 3734, 4765, 4257, 677, 2727, 4775, 5293, 3760, 1718, 3255, 4792, 185, 5308, 1728, 4289, 4805, 3782, 4817, 3282, 1235, 5338, 4827, 3292, 1245, 734, 3297, 3303, 2280, 3818, 1771, 1264, 3830, 2296, 764, 3838, 4359, 2317, 5390, 790, 5912, 5414, 4391, 813, 3374, 817, 4914, 2866, 4403, 1847, 1336, 5434, 4932, 2889, 2382, 1876, 5973, 856, 1883, 4451, 357, 1390, 4472, 4473, 891, 2429, 3458, 3463, 2444, 5519, 916, 3994, 927, 418, 4012, 4530, 953, 6079, 4544, 3525, 2502, 5575, 6096, 5584, 4050, 2002, 6105, 5082, 474, 6109, 3554, 4076, 4080, 505}, {1938}, {5614}, {3196}, {4595}, {1520}, {4883}, {2141}, {4225}]), (-0.03205862435538587, [{4099, 5, 1542, 11, 4621, 20, 532, 4632, 1048, 539, 3612, 29, 5155, 2603, 3121, 3634, 1077, 54, 2613, 5687, 2113, 578, 3143, 5193, 3577, 2639, 3668, 605, 5213, 95, 4708, 616, 4721, 5234, 5755, 638, 639, 4739, 656, 3734, 4765, 4257, 677, 2727, 4775, 5293, 3760, 1718, 3255, 4792, 185, 5308, 1728, 4289, 4805, 3782, 4817, 3282, 1235, 5338, 4827, 3292, 1245, 734, 3297, 3303, 2280, 3818, 1771, 1264, 3830, 2296, 764, 3838, 4359, 2317, 5390, 790, 5912, 5414, 4391, 813, 3374, 817, 4914, 2866, 4403, 1847, 1336, 5434, 4932, 2889, 2382, 1876, 5973, 856, 1883, 4451, 357, 1390, 4472, 4473, 891, 2429, 3458, 3463, 2444, 5519, 916, 3994, 927, 418, 4012, 4530, 953, 6079, 4544, 3525, 2502, 5575, 6096, 5584, 4050, 2002, 6105, 5082, 474, 6109, 3554, 4076, 4080, 505}, {2141}, {5614}, {4595}, {1520}, {3196}, {4883}, {1938}, {1643}, {4225}]), (-0.03621039204898091, [{2141}, {4099, 5, 1542, 11, 4621, 20, 532, 4632, 1048, 539, 3612, 29, 5155, 2603, 3121, 3634, 1077, 54, 2613, 5687, 2113, 578, 3143, 5193, 3577, 2639, 605, 5213, 95, 4708, 616, 4721, 5234, 5755, 638, 639, 4739, 656, 3734, 4765, 4257, 677, 2727, 4775, 5293, 3760, 1718, 3255, 4792, 185, 5308, 1728, 4289, 4805, 3782, 4817, 3282, 1235, 5338, 4827, 3292, 1245, 734, 3297, 3303, 2280, 3818, 1771, 1264, 3830, 2296, 764, 3838, 4359, 2317, 5390, 790, 5912, 5414, 4391, 813, 3374, 817, 4914, 2866, 4403, 1847, 1336, 5434, 4932, 2889, 2382, 1876, 5973, 856, 1883, 4451, 357, 1390, 4472, 4473, 891, 2429, 3458, 3463, 2444, 5519, 916, 3994, 927, 418, 4012, 4530, 953, 6079, 4544, 3525, 2502, 5575, 6096, 5584, 4050, 2002, 6105, 5082, 474, 6109, 3554, 4076, 4080, 505}, {1520}, {4883}, {1643}, {3196}, {5614}, {3668}, {1938}, {4595}, {4225}]), (-0.03976695674381392, [{4099, 5, 1542, 11, 4621, 20, 532, 4632, 1048, 539, 3612, 29, 5155, 2603, 3121, 3634, 1077, 54, 2613, 5687, 2113, 578, 3143, 5193, 3577, 2639, 605, 5213, 95, 4708, 616, 4721, 5234, 5755, 638, 639, 4739, 656, 3734, 4765, 4257, 677, 2727, 4775, 5293, 3760, 1718, 3255, 4792, 185, 5308, 1728, 4289, 4805, 3782, 4817, 3282, 1235, 5338, 4827, 3292, 1245, 734, 3297, 3303, 2280, 3818, 1771, 1264, 3830, 2296, 764, 3838, 4359, 2317, 5390, 790, 5912, 5414, 813, 3374, 817, 4914, 2866, 4403, 1847, 1336, 5434, 4932, 2889, 2382, 1876, 5973, 856, 1883, 4451, 357, 1390, 4472, 4473, 891, 2429, 3458, 3463, 2444, 5519, 916, 3994, 927, 418, 4012, 4530, 953, 6079, 4544, 3525, 2502, 5575, 6096, 5584, 4050, 2002, 6105, 5082, 474, 6109, 3554, 4076, 4080, 505}, {1643}, {3668}, {1520}, {2141}, {4391}, {4225}, {4595}, {1938}, {4883}, {5614}, {3196}]), (-0.04272901621596714, [{4099, 5, 1542, 11, 4621, 20, 532, 4632, 1048, 539, 3612, 29, 5155, 2603, 3121, 3634, 1077, 54, 2613, 5687, 2113, 578, 3143, 5193, 3577, 2639, 605, 5213, 95, 4708, 616, 4721, 5234, 5755, 638, 639, 4739, 656, 3734, 4765, 4257, 677, 2727, 4775, 5293, 3760, 1718, 3255, 185, 5308, 1728, 4289, 4805, 3782, 4817, 3282, 1235, 5338, 4827, 3292, 1245, 734, 3297, 3303, 2280, 3818, 1771, 1264, 3830, 2296, 764, 3838, 4359, 2317, 5390, 790, 5912, 5414, 813, 3374, 817, 4914, 2866, 4403, 1847, 1336, 5434, 4932, 2889, 2382, 1876, 5973, 856, 1883, 4451, 357, 1390, 4472, 4473, 891, 2429, 3458, 3463, 2444, 5519, 916, 3994, 927, 418, 4012, 4530, 953, 6079, 4544, 3525, 2502, 5575, 6096, 5584, 4050, 2002, 6105, 5082, 474, 6109, 3554, 4076, 4080, 505}, {4225}, {1520}, {5614}, {1643}, {4595}, {3668}, {4883}, {4391}, {2141}, {4792}, {1938}, {3196}]), (-0.04747668468440802, [{4099, 5, 1542, 11, 4621, 20, 532, 4632, 1048, 539, 3612, 29, 5155, 2603, 3121, 3634, 1077, 54, 2613, 5687, 2113, 578, 3143, 5193, 3577, 2639, 605, 5213, 95, 4708, 616, 4721, 5234, 5755, 638, 639, 4739, 656, 3734, 4765, 4257, 677, 2727, 4775, 5293, 3760, 1718, 3255, 185, 5308, 1728, 4289, 4805, 3782, 4817, 3282, 1235, 5338, 4827, 3292, 1245, 734, 3297, 3303, 2280, 3818, 1771, 1264, 3830, 2296, 764, 3838, 4359, 2317, 5390, 790, 5414, 813, 3374, 817, 4914, 2866, 4403, 1847, 1336, 5434, 4932, 2889, 2382, 1876, 5973, 856, 1883, 4451, 357, 1390, 4472, 4473, 891, 2429, 3458, 3463, 2444, 5519, 916, 3994, 927, 418, 4012, 4530, 953, 6079, 4544, 3525, 2502, 5575, 6096, 5584, 4050, 2002, 6105, 5082, 474, 6109, 3554, 4076, 4080, 505}, {5912}, {4595}, {1520}, {1938}, {4792}, {4391}, {4883}, {5614}, {4225}, {3668}, {2141}, {3196}, {1643}]), (0.12054012055477403, [{4099, 5, 1542, 532, 20, 4632, 1048, 539, 3612, 5155, 1077, 5687, 578, 3143, 605, 95, 4721, 5755, 638, 639, 4739, 656, 3734, 4257, 677, 2727, 4775, 5293, 1718, 3255, 185, 5308, 1728, 4289, 4805, 3782, 4817, 3282, 1235, 5338, 4827, 3292, 734, 3297, 3818, 1771, 3830, 2296, 3838, 4359, 2317, 5390, 790, 813, 3374, 817, 2866, 4403, 4932, 2889, 2382, 5973, 1883, 4451, 357, 1390, 4472, 4473, 891, 2429, 3463, 916, 418, 4012, 4530, 6079, 2502, 6096, 4050, 2002, 6105, 5082, 474, 6109, 3554, 4076, 4080}, {3458, 11, 2444, 4621, 5519, 3994, 4765, 29, 927, 5414, 2603, 3760, 3121, 3634, 4914, 2613, 54, 1847, 1336, 953, 5434, 4544, 2113, 3525, 5575, 5193, 3577, 2639, 5584, 1876, 856, 5213, 1245, 4708, 3303, 616, 2280, 1264, 5234, 505, 764}, {4883}, {4391}, {1520}, {1643}, {3668}, {4595}, {3196}, {4225}, {4792}, {1938}, {5912}, {2141}, {5614}]), (0.23004630790974545, [{4225}, {3458, 11, 2444, 4621, 5519, 3994, 4765, 29, 927, 5414, 2603, 3760, 3121, 3634, 4914, 2613, 54, 1847, 1336, 953, 5434, 4544, 2113, 3525, 5575, 5193, 3577, 2639, 5584, 1876, 856, 5213, 1245, 4708, 3303, 616, 2280, 1264, 5234, 505, 764}, {5, 4359, 2317, 656, 532, 790, 539, 4257, 4775, 4012, 5293, 813, 3374, 817, 4403, 1077, 185, 6079, 1728, 578, 4805, 3782, 3143, 2502, 2889, 1235, 5338, 4827, 734, 3297, 3554, 1771, 4076, 1390, 4721, 3830, 891, 2429, 3838}, {4099, 4739, 1542, 3463, 5390, 20, 916, 3734, 4632, 1048, 3612, 418, 5155, 677, 2727, 2866, 4530, 1718, 3255, 5687, 5308, 4289, 4932, 2382, 6096, 4817, 3282, 4050, 2002, 5973, 6105, 5082, 1883, 3292, 605, 474, 95, 6109, 4451, 4472, 357, 3818, 4080, 2296, 4473, 5755, 638, 639}, {3668}, {5614}, {4391}, {3196}, {2141}, {1938}, {5912}, {4883}, {4595}, {1520}, {1643}, {4792}]), (0.22934364739420754, [{3458, 11, 2444, 4621, 5519, 3994, 4765, 29, 927, 5414, 2603, 3760, 3121, 3634, 4914, 2613, 54, 1847, 1336, 953, 5434, 4544, 2113, 3525, 5575, 5193, 3577, 2639, 5584, 1876, 856, 5213, 1245, 4708, 3303, 616, 2280, 1264, 5234, 505, 764}, {5, 4359, 2317, 656, 532, 790, 539, 4257, 4775, 4012, 5293, 813, 3374, 817, 4403, 1077, 185, 6079, 1728, 578, 4805, 3782, 3143, 2502, 2889, 1235, 5338, 4827, 734, 3297, 3554, 1771, 4076, 1390, 4721, 3830, 891, 2429, 3838}, {4099, 4739, 1542, 3463, 5390, 20, 916, 3734, 4632, 1048, 3612, 418, 5155, 677, 2727, 2866, 4530, 1718, 3255, 5687, 5308, 4289, 4932, 2382, 6096, 4817, 3282, 4050, 2002, 5973, 6105, 5082, 1883, 3292, 605, 474, 95, 6109, 4451, 4472, 357, 3818, 2296, 4473, 5755, 638, 639}, {4595}, {4792}, {4225}, {2141}, {5614}, {4391}, {3668}, {5912}, {4080}, {4883}, {1938}, {1520}, {1643}, {3196}]), (0.22695236875791325, [{1938}, {4099, 4739, 1542, 3463, 5390, 20, 916, 3734, 4632, 1048, 3612, 418, 5155, 677, 2727, 2866, 4530, 1718, 3255, 5687, 4289, 4932, 2382, 6096, 4817, 3282, 4050, 2002, 5973, 6105, 5082, 1883, 3292, 605, 474, 95, 6109, 4451, 4472, 357, 3818, 2296, 4473, 5755, 638, 639}, {3458, 11, 2444, 4621, 5519, 3994, 4765, 29, 927, 5414, 2603, 3760, 3121, 3634, 4914, 2613, 54, 1847, 1336, 953, 5434, 4544, 2113, 3525, 5575, 5193, 3577, 2639, 5584, 1876, 856, 5213, 1245, 4708, 3303, 616, 2280, 1264, 5234, 505, 764}, {1643}, {5, 4359, 2317, 656, 532, 790, 539, 4257, 4775, 4012, 5293, 813, 3374, 817, 4403, 1077, 185, 6079, 1728, 578, 4805, 3782, 3143, 2502, 2889, 1235, 5338, 4827, 734, 3297, 3554, 1771, 4076, 1390, 4721, 3830, 891, 2429, 3838}, {4595}, {3196}, {4080}, {4792}, {5614}, {3668}, {4391}, {4883}, {1520}, {5912}, {4225}, {2141}, {5308}]), (0.22252916616804752, [{3458, 11, 2444, 4621, 5519, 3994, 4765, 29, 927, 5414, 2603, 3760, 3121, 3634, 4914, 2613, 54, 1847, 1336, 953, 5434, 4544, 2113, 3525, 5575, 5193, 3577, 2639, 5584, 1876, 856, 5213, 1245, 4708, 3303, 616, 2280, 1264, 5234, 505}, {5, 4359, 2317, 656, 532, 790, 539, 4257, 4775, 4012, 5293, 813, 3374, 817, 4403, 1077, 185, 6079, 1728, 578, 4805, 3782, 3143, 2502, 2889, 1235, 5338, 4827, 734, 3297, 3554, 1771, 4076, 1390, 4721, 3830, 891, 2429, 3838}, {4099, 4739, 1542, 3463, 5390, 20, 916, 3734, 4632, 1048, 3612, 418, 5155, 677, 2727, 2866, 4530, 1718, 3255, 5687, 4289, 4932, 2382, 6096, 4817, 3282, 4050, 2002, 5973, 6105, 5082, 1883, 3292, 605, 474, 95, 6109, 4451, 4472, 357, 3818, 2296, 4473, 5755, 638, 639}, {3196}, {5308}, {3668}, {5912}, {2141}, {4391}, {1643}, {4225}, {4080}, {4792}, {4595}, {1520}, {5614}, {764}, {1938}, {4883}]), (0.21989296812663517, [{4099, 4739, 1542, 3463, 5390, 20, 916, 3734, 4632, 1048, 3612, 418, 5155, 677, 2727, 2866, 4530, 1718, 3255, 5687, 4289, 4932, 2382, 6096, 4817, 3282, 4050, 2002, 5973, 6105, 5082, 1883, 3292, 605, 474, 95, 6109, 4451, 4472, 357, 3818, 2296, 4473, 5755, 638, 639}, {3458, 11, 2444, 4621, 5519, 3994, 4765, 29, 927, 5414, 2603, 3760, 3121, 3634, 4914, 2613, 54, 1847, 1336, 953, 5434, 4544, 2113, 3525, 5575, 5193, 3577, 2639, 5584, 1876, 856, 5213, 1245, 4708, 3303, 616, 2280, 1264, 5234, 505}, {734}, {5614}, {5, 4359, 2317, 656, 532, 790, 539, 4257, 4775, 4012, 5293, 813, 3374, 817, 4403, 1077, 185, 6079, 1728, 578, 4805, 3782, 3143, 2502, 2889, 1235, 5338, 4827, 3297, 3554, 1771, 4076, 1390, 4721, 3830, 891, 2429, 3838}, {4595}, {3668}, {5308}, {1520}, {5912}, {3196}, {4391}, {4792}, {4225}, {764}, {2141}, {4080}, {1643}, {1938}, {4883}]), (0.21336178399015554, [{5, 4359, 2317, 656, 532, 790, 539, 4257, 4775, 4012, 5293, 813, 3374, 817, 4403, 185, 6079, 578, 4805, 3782, 3143, 2889, 1235, 5338, 3297, 3554, 1771, 4076, 1390, 4721, 3830, 891, 2429, 3838}, {4099, 4739, 1542, 3463, 5390, 20, 916, 3734, 4632, 1048, 3612, 418, 5155, 677, 2727, 2866, 4530, 1718, 3255, 5687, 4289, 4932, 2382, 6096, 4817, 3282, 4050, 2002, 5973, 6105, 5082, 1883, 3292, 605, 474, 95, 6109, 4451, 4472, 357, 3818, 2296, 4473, 5755, 638, 639}, {3458, 11, 2444, 4621, 5519, 3994, 4765, 29, 927, 5414, 2603, 3760, 3121, 3634, 4914, 2613, 54, 1847, 1336, 953, 5434, 4544, 2113, 3525, 5575, 5193, 3577, 2639, 5584, 1876, 856, 5213, 1245, 4708, 3303, 616, 2280, 1264, 5234, 505}, {5614}, {1728, 4827, 1077, 2502}, {1938}, {4883}, {3196}, {4595}, {764}, {2141}, {5308}, {4792}, {734}, {1643}, {4391}, {3668}, {1520}, {4225}, {5912}, {4080}]), (0.2112063536699007, [{5308}, {4099, 4739, 1542, 3463, 5390, 20, 916, 3734, 4632, 1048, 3612, 418, 5155, 677, 2727, 2866, 4530, 1718, 3255, 5687, 4289, 4932, 2382, 6096, 4817, 3282, 4050, 2002, 5973, 6105, 5082, 1883, 3292, 605, 474, 95, 6109, 4451, 4472, 357, 3818, 2296, 4473, 5755, 638, 639}, {5, 4359, 2317, 656, 790, 539, 4257, 4775, 4012, 5293, 813, 3374, 817, 4403, 185, 6079, 578, 4805, 3782, 3143, 2889, 1235, 5338, 3297, 3554, 1771, 4076, 1390, 4721, 3830, 891, 2429, 3838}, {532}, {4080}, {1520}, {3458, 11, 2444, 4621, 5519, 3994, 4765, 29, 927, 5414, 2603, 3760, 3121, 3634, 4914, 2613, 54, 1847, 1336, 953, 5434, 4544, 2113, 3525, 5575, 5193, 3577, 2639, 5584, 1876, 856, 5213, 1245, 4708, 3303, 616, 2280, 1264, 5234, 505}, {3668}, {734}, {1938}, {4883}, {4391}, {1728, 4827, 1077, 2502}, {3196}, {4792}, {5912}, {1643}, {5614}, {764}, {2141}, {4595}, {4225}]), (0.20818568100677898, [{4099, 4739, 1542, 3463, 5390, 20, 916, 3734, 4632, 1048, 3612, 418, 5155, 677, 2727, 2866, 4530, 1718, 3255, 5687, 4289, 4932, 2382, 6096, 4817, 3282, 4050, 2002, 5973, 6105, 5082, 1883, 3292, 605, 474, 95, 6109, 4451, 4472, 357, 3818, 2296, 4473, 5755, 638, 639}, {5, 4359, 2317, 656, 790, 539, 4257, 4775, 4012, 5293, 813, 3374, 817, 4403, 185, 6079, 578, 4805, 3782, 3143, 2889, 1235, 5338, 3297, 1771, 4076, 1390, 4721, 3830, 891, 2429, 3838}, {5614}, {3458, 11, 2444, 4621, 5519, 3994, 4765, 29, 927, 5414, 2603, 3760, 3121, 3634, 4914, 2613, 54, 1847, 1336, 953, 5434, 4544, 2113, 3525, 5575, 5193, 3577, 2639, 5584, 1876, 856, 5213, 1245, 4708, 3303, 616, 2280, 1264, 5234, 505}, {4792}, {4595}, {3554}, {1728, 4827, 1077, 2502}, {3196}, {4225}, {1938}, {4080}, {3668}, {1643}, {764}, {532}, {5912}, {5308}, {2141}, {4883}, {4391}, {1520}, {734}]), (0.20686339532957526, [{4099, 4739, 1542, 3463, 20, 916, 3734, 4632, 1048, 3612, 418, 5155, 677, 2727, 2866, 4530, 1718, 3255, 5687, 4289, 4932, 2382, 6096, 4817, 3282, 4050, 2002, 5973, 6105, 5082, 1883, 3292, 605, 474, 95, 6109, 4451, 4472, 357, 3818, 2296, 4473, 5755, 638, 639}, {5, 4359, 2317, 656, 790, 539, 4257, 4775, 4012, 5293, 813, 3374, 817, 4403, 185, 6079, 578, 4805, 3782, 3143, 2889, 1235, 5338, 3297, 1771, 4076, 1390, 4721, 3830, 891, 2429, 3838}, {734}, {764}, {3458, 11, 2444, 4621, 5519, 3994, 4765, 29, 927, 5414, 2603, 3760, 3121, 3634, 4914, 2613, 54, 1847, 1336, 953, 5434, 4544, 2113, 3525, 5575, 5193, 3577, 2639, 5584, 1876, 856, 5213, 1245, 4708, 3303, 616, 2280, 1264, 5234, 505}, {1938}, {5308}, {4391}, {1728, 4827, 1077, 2502}, {3668}, {5390}, {5614}, {3554}, {2141}, {4883}, {4225}, {1520}, {3196}, {4792}, {4595}, {1643}, {4080}, {5912}, {532}]), (0.20621027691592736, [{3458, 2444, 4621, 5519, 3994, 4765, 29, 927, 5414, 2603, 3760, 3121, 3634, 4914, 2613, 54, 1847, 1336, 953, 5434, 4544, 2113, 3525, 5575, 5193, 3577, 2639, 5584, 1876, 856, 5213, 1245, 4708, 3303, 616, 2280, 1264, 5234, 505}, {5, 4359, 2317, 656, 790, 539, 4257, 4775, 4012, 5293, 813, 3374, 817, 4403, 185, 6079, 578, 4805, 3782, 3143, 2889, 1235, 5338, 3297, 1771, 4076, 1390, 4721, 3830, 891, 2429, 3838}, {11}, {4099, 4739, 1542, 3463, 20, 916, 3734, 4632, 1048, 3612, 418, 5155, 677, 2727, 2866, 4530, 1718, 3255, 5687, 4289, 4932, 2382, 6096, 4817, 3282, 4050, 2002, 5973, 6105, 5082, 1883, 3292, 605, 474, 95, 6109, 4451, 4472, 357, 3818, 2296, 4473, 5755, 638, 639}, {1643}, {4391}, {4792}, {4225}, {1938}, {764}, {734}, {1728, 4827, 1077, 2502}, {3668}, {3196}, {2141}, {5390}, {4595}, {5614}, {3554}, {1520}, {4883}, {5912}, {5308}, {532}, {4080}]), (0.20117303137305925, [{4099, 4739, 1542, 3463, 20, 916, 3734, 4632, 1048, 3612, 418, 5155, 677, 2727, 2866, 4530, 1718, 3255, 5687, 4289, 4932, 2382, 6096, 4817, 3282, 4050, 2002, 5973, 6105, 5082, 1883, 3292, 605, 474, 95, 6109, 4451, 4472, 357, 3818, 2296, 4473, 5755, 638, 639}, {5, 4359, 2317, 656, 790, 539, 4257, 4775, 4012, 5293, 813, 3374, 817, 4403, 185, 6079, 578, 4805, 3782, 3143, 2889, 1235, 5338, 3297, 1771, 4076, 1390, 4721, 3830, 891, 2429, 3838}, {3458, 2444, 4621, 5519, 3994, 4765, 29, 927, 5414, 2603, 3760, 3121, 3634, 4914, 2613, 54, 1847, 1336, 953, 5434, 4544, 2113, 3525, 5575, 5193, 2639, 5584, 1876, 856, 5213, 1245, 4708, 3303, 616, 2280, 1264, 5234, 3577}, {3196}, {1728, 4827, 1077, 2502}, {2141}, {3554}, {4595}, {4225}, {734}, {11}, {4391}, {4883}, {5390}, {3668}, {5912}, {505}, {764}, {1520}, {4080}, {532}, {1643}, {1938}, {5308}, {4792}, {5614}]), (0.19777974628163836, [{3458, 2444, 4621, 5519, 3994, 4765, 29, 927, 5414, 2603, 3760, 3121, 3634, 4914, 2613, 54, 1847, 1336, 953, 5434, 4544, 2113, 3525, 5575, 5193, 2639, 5584, 856, 5213, 1245, 4708, 3303, 616, 2280, 1264, 5234, 3577}, {4099, 4739, 1542, 3463, 20, 916, 3734, 4632, 1048, 3612, 418, 5155, 677, 2727, 2866, 4530, 1718, 3255, 5687, 4289, 4932, 2382, 6096, 4817, 3282, 4050, 2002, 5973, 6105, 5082, 1883, 3292, 605, 474, 95, 6109, 4451, 4472, 357, 3818, 2296, 4473, 5755, 638, 639}, {5614}, {5, 4359, 2317, 656, 790, 539, 4257, 4775, 4012, 5293, 813, 3374, 817, 4403, 185, 6079, 578, 4805, 3782, 3143, 2889, 1235, 5338, 3297, 1771, 4076, 1390, 4721, 3830, 891, 2429, 3838}, {4792}, {4595}, {5912}, {2141}, {4225}, {1728, 4827, 1077, 2502}, {1520}, {734}, {3196}, {4391}, {1643}, {1876}, {4883}, {3668}, {5390}, {505}, {3554}, {5308}, {4080}, {1938}, {11}, {532}, {764}]), (0.19768066207785862, [{4099, 4739, 1542, 3463, 20, 916, 3734, 4632, 1048, 3612, 418, 5155, 677, 2727, 2866, 4530, 1718, 3255, 5687, 4289, 4932, 2382, 6096, 4817, 3282, 4050, 2002, 5973, 6105, 5082, 1883, 3292, 605, 474, 95, 6109, 4451, 4472, 357, 3818, 2296, 4473, 5755, 638, 639}, {3458, 2444, 4621, 5519, 3994, 4765, 29, 927, 5414, 2603, 3760, 3121, 3634, 4914, 2613, 54, 1847, 1336, 953, 5434, 4544, 2113, 3525, 5575, 5193, 2639, 5584, 856, 5213, 1245, 4708, 3303, 616, 2280, 1264, 5234, 3577}, {5, 4359, 2317, 656, 790, 539, 4257, 4775, 4012, 5293, 813, 3374, 817, 4403, 185, 6079, 578, 4805, 3782, 3143, 2889, 1235, 5338, 3297, 4076, 1390, 4721, 3830, 891, 2429, 3838}, {1520}, {5308}, {532}, {5912}, {2141}, {1771}, {505}, {1876}, {5390}, {1728, 4827, 1077, 2502}, {4391}, {3196}, {1643}, {4792}, {764}, {1938}, {5614}, {4595}, {11}, {3554}, {4883}, {3668}, {4225}, {734}, {4080}]), (0.19248292803591038, [{5, 4359, 2317, 656, 790, 539, 4257, 4775, 4012, 5293, 813, 3374, 817, 4403, 185, 6079, 578, 4805, 3782, 3143, 2889, 1235, 5338, 3297, 4076, 1390, 4721, 3830, 891, 2429, 3838}, {3458, 2444, 4621, 5519, 3994, 4765, 29, 927, 5414, 2603, 3760, 3121, 3634, 4914, 2613, 54, 1847, 1336, 953, 5434, 4544, 2113, 3525, 5575, 5193, 2639, 5584, 856, 5213, 1245, 4708, 3303, 616, 2280, 5234, 3577}, {1264}, {4099, 4739, 1542, 3463, 20, 916, 3734, 4632, 1048, 3612, 418, 5155, 677, 2727, 2866, 4530, 1718, 3255, 5687, 4289, 4932, 2382, 6096, 4817, 3282, 4050, 2002, 5973, 6105, 5082, 1883, 3292, 605, 474, 95, 6109, 4451, 4472, 357, 3818, 2296, 4473, 5755, 638, 639}, {3668}, {4883}, {5912}, {1728, 4827, 1077, 2502}, {1876}, {1643}, {1520}, {532}, {1771}, {4080}, {3554}, {764}, {5308}, {4792}, {4225}, {734}, {2141}, {4391}, {1938}, {5390}, {505}, {3196}, {5614}, {4595}, {11}]), (0.19060102594017492, [{3458, 2444, 4621, 5519, 3994, 4765, 29, 927, 5414, 2603, 3760, 3121, 3634, 4914, 2613, 54, 1847, 1336, 953, 5434, 4544, 2113, 3525, 5575, 5193, 2639, 5584, 856, 5213, 1245, 4708, 3303, 616, 2280, 5234, 3577}, {5, 4359, 2317, 656, 790, 539, 4257, 4775, 4012, 5293, 813, 3374, 817, 4403, 185, 6079, 578, 4805, 3782, 3143, 2889, 1235, 5338, 3297, 4076, 1390, 4721, 3830, 891, 3838}, {4099, 4739, 1542, 3463, 20, 916, 3734, 4632, 1048, 3612, 418, 5155, 677, 2727, 2866, 4530, 1718, 3255, 5687, 4289, 4932, 2382, 6096, 4817, 3282, 4050, 2002, 5973, 6105, 5082, 1883, 3292, 605, 474, 95, 6109, 4451, 4472, 357, 3818, 2296, 4473, 5755, 638, 639}, {3196}, {4225}, {1876}, {11}, {764}, {5912}, {505}, {5390}, {4080}, {1728, 4827, 1077, 2502}, {2141}, {2429}, {1938}, {3554}, {1264}, {5614}, {4792}, {4883}, {4391}, {5308}, {1771}, {532}, {4595}, {3668}, {1520}, {1643}, {734}]), (0.18694398148939617, [{3458, 2444, 4621, 5519, 3994, 4765, 29, 927, 5414, 2603, 3760, 3121, 3634, 4914, 2613, 54, 1847, 1336, 953, 5434, 4544, 2113, 3525, 5575, 5193, 2639, 5584, 856, 5213, 1245, 4708, 3303, 616, 2280, 5234, 3577}, {4099, 4739, 1542, 3463, 20, 916, 3734, 4632, 1048, 3612, 418, 5155, 677, 2727, 2866, 4530, 1718, 3255, 5687, 4289, 4932, 2382, 6096, 4817, 3282, 4050, 5973, 6105, 5082, 1883, 3292, 605, 474, 95, 6109, 4451, 4472, 357, 3818, 2296, 4473, 5755, 638, 639}, {5912}, {5, 4359, 2317, 656, 790, 539, 4257, 4775, 4012, 5293, 813, 3374, 817, 4403, 185, 6079, 578, 4805, 3782, 3143, 2889, 1235, 5338, 3297, 4076, 1390, 4721, 3830, 891, 3838}, {734}, {3554}, {5614}, {3196}, {1264}, {4792}, {2002}, {1728, 4827, 1077, 2502}, {764}, {1520}, {5390}, {532}, {4225}, {4080}, {505}, {1876}, {1771}, {5308}, {1938}, {3668}, {4595}, {1643}, {2141}, {11}, {2429}, {4883}, {4391}]), (0.1844257076060034, [{1520}, {3196}, {3458, 2444, 4621, 5519, 3994, 4765, 29, 5414, 2603, 3760, 3121, 3634, 4914, 2613, 54, 1847, 1336, 953, 5434, 4544, 2113, 3525, 5575, 5193, 2639, 5584, 856, 5213, 1245, 4708, 3303, 616, 2280, 5234, 3577}, {5, 4359, 2317, 656, 790, 539, 4257, 4775, 4012, 5293, 813, 3374, 817, 4403, 185, 6079, 578, 4805, 3782, 3143, 2889, 1235, 5338, 3297, 4076, 1390, 4721, 3830, 891, 3838}, {1938}, {4099, 4739, 1542, 3463, 20, 916, 3734, 4632, 1048, 3612, 418, 5155, 677, 2727, 2866, 4530, 1718, 3255, 5687, 4289, 4932, 2382, 6096, 4817, 3282, 4050, 5973, 6105, 5082, 1883, 3292, 605, 474, 95, 6109, 4451, 4472, 357, 3818, 2296, 4473, 5755, 638, 639}, {1264}, {1771}, {5308}, {1728, 4827, 1077, 2502}, {2002}, {4391}, {3668}, {1643}, {927}, {734}, {2429}, {4080}, {4792}, {1876}, {5614}, {505}, {3554}, {2141}, {5912}, {5390}, {532}, {4225}, {4883}, {764}, {4595}, {11}]), (0.17960477264885227, [{4099, 4739, 1542, 3463, 20, 916, 3734, 4632, 1048, 3612, 418, 5155, 677, 2727, 2866, 4530, 1718, 3255, 5687, 4932, 2382, 6096, 4817, 3282, 4050, 5973, 6105, 5082, 1883, 3292, 605, 474, 95, 6109, 4451, 4472, 357, 3818, 2296, 4473, 5755, 638, 639}, {505}, {3458, 2444, 4621, 5519, 3994, 4765, 29, 5414, 2603, 3760, 3121, 3634, 4914, 2613, 54, 1847, 1336, 953, 5434, 4544, 2113, 3525, 5575, 5193, 2639, 5584, 856, 5213, 1245, 4708, 3303, 616, 2280, 5234, 3577}, {5, 4359, 2317, 656, 790, 539, 4257, 4775, 4012, 5293, 813, 3374, 817, 4403, 185, 6079, 578, 4805, 3782, 3143, 2889, 1235, 5338, 3297, 4076, 1390, 4721, 3830, 891, 3838}, {1771}, {5390}, {5614}, {1728, 4827, 1077, 2502}, {2141}, {3554}, {4883}, {764}, {11}, {4792}, {1938}, {5308}, {927}, {1643}, {4595}, {4391}, {2429}, {1876}, {3668}, {3196}, {1520}, {532}, {5912}, {734}, {4225}, {2002}, {1264}, {4080}, {4289}]), (0.17655618894241215, [{5308}, {4099, 4739, 1542, 3463, 20, 916, 3734, 4632, 1048, 3612, 418, 5155, 677, 2727, 2866, 4530, 1718, 3255, 5687, 4932, 2382, 6096, 4817, 3282, 4050, 5973, 6105, 5082, 1883, 3292, 605, 474, 95, 6109, 4451, 4472, 357, 3818, 2296, 4473, 5755, 638, 639}, {3458, 2444, 4621, 5519, 3994, 4765, 29, 5414, 2603, 3760, 3121, 3634, 4914, 2613, 54, 1847, 1336, 953, 5434, 4544, 2113, 3525, 5575, 5193, 2639, 5584, 856, 5213, 1245, 4708, 3303, 616, 2280, 5234, 3577}, {5, 4359, 2317, 656, 790, 539, 4257, 4775, 4012, 5293, 813, 3374, 817, 4403, 185, 6079, 578, 4805, 3782, 3143, 1235, 5338, 3297, 4076, 1390, 4721, 3830, 891, 3838}, {532}, {5390}, {1938}, {764}, {11}, {4595}, {3668}, {3196}, {3554}, {505}, {4080}, {1728, 4827, 1077, 2502}, {4391}, {1876}, {1771}, {2429}, {5912}, {1643}, {4883}, {4289}, {734}, {1264}, {1520}, {2141}, {2889}, {5614}, {2002}, {4225}, {927}, {4792}]), (0.1729814820694225, [{3458, 2444, 4621, 5519, 3994, 4765, 29, 5414, 2603, 3760, 3121, 3634, 4914, 2613, 54, 1847, 953, 5434, 4544, 2113, 3525, 5575, 5193, 2639, 5584, 856, 5213, 1245, 4708, 3303, 616, 2280, 5234, 3577}, {5, 4359, 2317, 656, 790, 539, 4257, 4775, 4012, 5293, 813, 3374, 817, 4403, 185, 6079, 578, 4805, 3782, 3143, 1235, 5338, 3297, 4076, 1390, 4721, 3830, 891, 3838}, {11}, {4099, 4739, 1542, 3463, 20, 916, 3734, 4632, 1048, 3612, 418, 5155, 677, 2727, 2866, 4530, 1718, 3255, 5687, 4932, 2382, 6096, 4817, 3282, 4050, 5973, 6105, 5082, 1883, 3292, 605, 474, 95, 6109, 4451, 4472, 357, 3818, 2296, 4473, 5755, 638, 639}, {2889}, {4883}, {4391}, {3554}, {2002}, {1728, 4827, 1077, 2502}, {1771}, {927}, {532}, {4289}, {1643}, {5912}, {1520}, {1876}, {4225}, {1938}, {5390}, {3196}, {764}, {4595}, {1336}, {2141}, {5308}, {1264}, {4792}, {4080}, {734}, {5614}, {3668}, {2429}, {505}]), (0.16811449389079608, [{5, 4359, 2317, 656, 790, 539, 4257, 4775, 4012, 5293, 813, 3374, 817, 4403, 185, 6079, 578, 4805, 3782, 3143, 1235, 5338, 3297, 4076, 1390, 4721, 3830, 891, 3838}, {3668}, {3458, 2444, 4621, 5519, 3994, 4765, 29, 5414, 2603, 3760, 3121, 3634, 4914, 2613, 54, 1847, 953, 5434, 4544, 2113, 3525, 5575, 5193, 2639, 5584, 856, 5213, 1245, 4708, 3303, 616, 2280, 5234, 3577}, {4099, 4739, 1542, 3463, 20, 916, 3734, 4632, 1048, 3612, 418, 5155, 677, 2866, 4530, 1718, 3255, 5687, 4932, 2382, 6096, 4817, 3282, 4050, 5973, 6105, 5082, 1883, 3292, 605, 474, 95, 6109, 4451, 4472, 357, 3818, 2296, 4473, 5755, 638, 639}, {5912}, {4225}, {4883}, {1728, 4827, 1077, 2502}, {4080}, {1336}, {5308}, {2889}, {2141}, {505}, {4595}, {1938}, {1771}, {764}, {4289}, {3554}, {532}, {2002}, {5614}, {5390}, {1264}, {4792}, {1876}, {2429}, {1643}, {2727}, {4391}, {11}, {927}, {1520}, {734}, {3196}]), (0.16626817837529112, [{5, 4359, 2317, 656, 790, 539, 4257, 4775, 4012, 5293, 813, 3374, 817, 4403, 185, 6079, 578, 4805, 3782, 3143, 1235, 5338, 3297, 4076, 1390, 4721, 3830, 3838}, {3196}, {1336}, {4099, 4739, 1542, 3463, 20, 916, 3734, 4632, 1048, 3612, 418, 5155, 677, 2866, 4530, 1718, 3255, 5687, 4932, 2382, 6096, 4817, 3282, 4050, 5973, 6105, 5082, 1883, 3292, 605, 474, 95, 6109, 4451, 4472, 357, 3818, 2296, 4473, 5755, 638, 639}, {3458, 2444, 4621, 5519, 3994, 4765, 29, 5414, 2603, 3760, 3121, 3634, 4914, 2613, 54, 1847, 953, 5434, 4544, 2113, 3525, 5575, 5193, 2639, 5584, 856, 5213, 1245, 4708, 3303, 616, 2280, 5234, 3577}, {1728, 4827, 1077, 2502}, {1771}, {2429}, {3554}, {4391}, {505}, {532}, {11}, {1264}, {4792}, {2889}, {4080}, {4225}, {5614}, {764}, {927}, {734}, {4595}, {1938}, {2141}, {1643}, {3668}, {1520}, {5912}, {891}, {1876}, {2727}, {4883}, {5308}, {5390}, {2002}, {4289}]), (0.1615644698000765, [{3458, 2444, 4621, 5519, 3994, 4765, 29, 5414, 2603, 3760, 3121, 3634, 4914, 2613, 54, 1847, 953, 5434, 4544, 2113, 3525, 5575, 5193, 2639, 5584, 856, 5213, 1245, 4708, 3303, 616, 2280, 5234, 3577}, {4099, 4739, 1542, 3463, 20, 916, 3734, 4632, 1048, 3612, 418, 5155, 677, 2866, 4530, 1718, 3255, 5687, 4932, 2382, 6096, 4817, 3282, 4050, 5973, 6105, 5082, 1883, 3292, 605, 474, 95, 6109, 4451, 4472, 357, 3818, 2296, 4473, 5755, 638, 639}, {5, 4359, 2317, 656, 790, 539, 4257, 4775, 4012, 5293, 813, 3374, 817, 185, 6079, 578, 4805, 3782, 3143, 1235, 5338, 3297, 4076, 1390, 4721, 3830, 3838}, {2141}, {3196}, {2889}, {3554}, {4792}, {4403}, {5912}, {1771}, {3668}, {1728, 4827, 1077, 2502}, {11}, {734}, {4080}, {764}, {5614}, {4289}, {2429}, {1520}, {927}, {4595}, {2002}, {1643}, {4883}, {5308}, {1336}, {2727}, {5390}, {1938}, {1264}, {1876}, {891}, {532}, {4391}, {505}, {4225}]), (0.1599414426311181, [{3458, 2444, 4621, 5519, 3994, 4765, 29, 5414, 2603, 3760, 3121, 3634, 4914, 2613, 54, 1847, 953, 5434, 4544, 2113, 3525, 5575, 5193, 2639, 5584, 856, 5213, 1245, 4708, 3303, 616, 2280, 5234, 3577}, {4099, 4739, 1542, 3463, 20, 916, 3734, 4632, 1048, 3612, 418, 5155, 677, 2866, 4530, 1718, 3255, 5687, 4932, 2382, 6096, 4817, 3282, 4050, 5973, 6105, 5082, 1883, 3292, 605, 474, 95, 6109, 4451, 4472, 357, 3818, 2296, 4473, 5755, 638, 639}, {5, 4359, 2317, 656, 790, 539, 4257, 4775, 4012, 5293, 813, 3374, 817, 6079, 578, 4805, 3782, 3143, 1235, 5338, 3297, 4076, 1390, 4721, 3830, 3838}, {4225}, {1728, 4827, 1077, 2502}, {5308}, {1643}, {2727}, {4595}, {1771}, {3196}, {1520}, {3668}, {4391}, {5614}, {505}, {4403}, {1264}, {891}, {2429}, {532}, {1938}, {927}, {3554}, {5390}, {1876}, {2002}, {4289}, {185}, {11}, {734}, {4080}, {2889}, {764}, {2141}, {4883}, {5912}, {4792}, {1336}]), (0.15591806573678696, [{5, 4359, 2317, 656, 790, 539, 4257, 4775, 4012, 5293, 813, 3374, 817, 6079, 578, 4805, 3782, 3143, 1235, 5338, 3297, 4076, 1390, 4721, 3830, 3838}, {3458, 2444, 4621, 5519, 3994, 4765, 29, 5414, 2603, 3760, 3121, 3634, 4914, 2613, 54, 1847, 953, 5434, 4544, 2113, 3525, 5575, 5193, 2639, 5584, 856, 5213, 4708, 3303, 616, 2280, 5234, 3577}, {4099, 4739, 1542, 3463, 20, 916, 3734, 4632, 1048, 3612, 418, 5155, 677, 2866, 4530, 1718, 3255, 5687, 4932, 2382, 6096, 4817, 3282, 4050, 5973, 6105, 5082, 1883, 3292, 605, 474, 95, 6109, 4451, 4472, 357, 3818, 2296, 4473, 5755, 638, 639}, {1938}, {505}, {2429}, {11}, {2141}, {4792}, {185}, {1728, 4827, 1077, 2502}, {4391}, {1336}, {1264}, {5912}, {5614}, {4289}, {4595}, {4883}, {1245}, {764}, {3668}, {3196}, {5390}, {2727}, {2002}, {1520}, {3554}, {4080}, {1771}, {4225}, {1643}, {2889}, {734}, {4403}, {891}, {1876}, {532}, {5308}, {927}]), (0.1499765023904065, [{5912}, {2889}, {1938}, {4099, 4739, 1542, 3463, 20, 916, 3734, 4632, 1048, 3612, 418, 5155, 677, 2866, 4530, 1718, 3255, 5687, 4932, 2382, 6096, 4817, 3282, 4050, 5973, 6105, 5082, 1883, 3292, 605, 474, 95, 6109, 4451, 4472, 357, 3818, 2296, 4473, 5755, 638, 639}, {1728, 4827, 1077, 2502}, {1264}, {1245}, {5, 4359, 2317, 656, 790, 539, 4257, 4775, 4012, 5293, 813, 3374, 817, 6079, 578, 4805, 3782, 3143, 1235, 5338, 3297, 4076, 1390, 4721, 3830, 3838}, {891}, {2429}, {4595}, {2002}, {3458, 2444, 4621, 3994, 4765, 29, 5414, 2603, 3760, 3121, 3634, 4914, 2613, 54, 1847, 953, 5434, 4544, 2113, 3525, 5575, 5193, 2639, 5584, 856, 5213, 4708, 3303, 616, 2280, 5234, 3577}, {3554}, {1520}, {4080}, {3196}, {185}, {927}, {1771}, {4792}, {11}, {2727}, {5390}, {1876}, {1643}, {2141}, {505}, {4289}, {4391}, {4403}, {532}, {734}, {764}, {3668}, {5519}, {5308}, {5614}, {4225}, {4883}, {1336}]), (0.14283825306175418, [{5, 4359, 2317, 656, 790, 539, 4257, 4775, 4012, 5293, 813, 3374, 817, 6079, 578, 4805, 3782, 3143, 1235, 5338, 3297, 4076, 1390, 4721, 3830, 3838}, {3458, 2444, 4621, 3994, 4765, 29, 5414, 2603, 3760, 3121, 3634, 4914, 2613, 54, 1847, 953, 5434, 4544, 2113, 3525, 5575, 5193, 2639, 5584, 856, 5213, 4708, 3303, 616, 2280, 5234, 3577}, {4099, 4739, 1542, 3463, 20, 916, 3734, 4632, 1048, 3612, 418, 5155, 677, 2866, 4530, 1718, 3255, 5687, 4932, 2382, 6096, 4817, 4050, 5973, 6105, 5082, 1883, 3292, 605, 474, 95, 6109, 4451, 4472, 357, 3818, 2296, 4473, 5755, 638, 639}, {1728, 4827, 1077, 2502}, {4289}, {1245}, {2002}, {532}, {1876}, {4080}, {4595}, {3282}, {1336}, {4403}, {2141}, {4225}, {1643}, {5308}, {3668}, {4883}, {3196}, {185}, {505}, {5614}, {2727}, {1264}, {734}, {2429}, {764}, {5390}, {891}, {4792}, {927}, {5519}, {11}, {2889}, {1938}, {1520}, {5912}, {3554}, {4391}, {1771}]), (0.13995783339130696, [{4099, 4739, 1542, 3463, 20, 916, 3734, 4632, 1048, 3612, 418, 5155, 677, 2866, 4530, 1718, 3255, 5687, 4932, 2382, 6096, 4817, 4050, 5973, 6105, 5082, 1883, 3292, 605, 474, 95, 6109, 4451, 4472, 357, 3818, 2296, 4473, 5755, 638, 639}, {2141}, {3282}, {734}, {11}, {3458, 2444, 4621, 3994, 4765, 29, 5414, 2603, 3760, 3121, 3634, 4914, 2613, 54, 1847, 953, 5434, 4544, 2113, 3525, 5575, 5193, 2639, 5584, 856, 5213, 4708, 3303, 616, 2280, 5234, 3577}, {4792}, {5519}, {2727}, {1643}, {4595}, {4080}, {3668}, {5, 4359, 2317, 790, 539, 4257, 4775, 4012, 5293, 813, 3374, 817, 6079, 578, 4805, 3782, 3143, 1235, 5338, 3297, 4076, 1390, 4721, 3830, 3838}, {532}, {4403}, {5390}, {5308}, {4391}, {1336}, {2889}, {1264}, {3196}, {1728, 4827, 1077, 2502}, {1938}, {4225}, {5912}, {2429}, {4289}, {1245}, {4883}, {5614}, {1771}, {891}, {3554}, {1520}, {505}, {2002}, {185}, {764}, {656}, {927}, {1876}]), (0.14032835249135706, [{3458, 2444, 4621, 3994, 4765, 29, 5414, 2603, 3760, 3121, 3634, 4914, 2613, 54, 1847, 953, 5434, 4544, 2113, 3525, 5575, 5193, 2639, 5584, 856, 5213, 4708, 3303, 616, 2280, 5234, 3577}, {5, 4359, 2317, 790, 539, 4257, 4775, 4012, 5293, 813, 3374, 817, 6079, 578, 4805, 3782, 3143, 1235, 5338, 3297, 4076, 1390, 4721, 3830, 3838}, {11}, {1520}, {4391}, {4099, 4739, 1542, 3463, 20, 916, 3734, 4632, 1048, 3612, 418, 5155, 677, 2866, 4530, 1718, 5687, 4932, 2382, 6096, 4817, 4050, 5973, 6105, 5082, 1883, 3292, 605, 474, 95, 6109, 4451, 4472, 357, 3818, 2296, 4473, 5755, 638, 639}, {3196}, {1876}, {5308}, {2727}, {3668}, {1728, 4827, 1077, 2502}, {1771}, {927}, {1264}, {4792}, {5390}, {2889}, {764}, {532}, {656}, {1938}, {505}, {1643}, {185}, {5519}, {4225}, {2141}, {5912}, {4289}, {4883}, {5614}, {3554}, {2429}, {734}, {1336}, {2002}, {1245}, {891}, {4403}, {3282}, {4595}, {4080}, {3255}]), (0.13713184025532993, [{4099, 4739, 1542, 3463, 20, 916, 3734, 4632, 1048, 3612, 418, 5155, 677, 2866, 4530, 1718, 5687, 4932, 2382, 6096, 4817, 4050, 5973, 6105, 5082, 1883, 3292, 605, 474, 95, 6109, 4451, 4472, 357, 3818, 2296, 4473, 5755, 638, 639}, {5, 4359, 2317, 790, 539, 4257, 4775, 4012, 5293, 813, 3374, 817, 6079, 578, 4805, 3782, 3143, 1235, 5338, 3297, 4076, 1390, 4721, 3830, 3838}, {3196}, {3458, 2444, 4621, 3994, 4765, 29, 5414, 2603, 3760, 3634, 4914, 2613, 54, 1847, 953, 5434, 4544, 2113, 3525, 5575, 5193, 2639, 5584, 856, 5213, 4708, 3303, 616, 2280, 5234, 3577}, {734}, {1771}, {1728, 4827, 1077, 2502}, {4403}, {1938}, {1336}, {764}, {505}, {5912}, {532}, {2002}, {891}, {1520}, {4225}, {927}, {4792}, {4289}, {2429}, {3282}, {1264}, {4883}, {11}, {4391}, {2727}, {4595}, {5614}, {5519}, {1245}, {3668}, {185}, {3121}, {2141}, {5308}, {5390}, {3554}, {656}, {1643}, {1876}, {2889}, {4080}, {3255}]), (0.1339269547063075, [{4099, 4739, 1542, 3463, 20, 916, 3734, 4632, 1048, 3612, 418, 5155, 677, 2866, 4530, 1718, 5687, 4932, 2382, 6096, 4817, 4050, 5973, 6105, 5082, 1883, 3292, 605, 474, 95, 6109, 4451, 4472, 357, 3818, 2296, 4473, 5755, 638, 639}, {1876}, {3458, 2444, 4621, 3994, 4765, 29, 5414, 2603, 3760, 3634, 4914, 2613, 54, 1847, 953, 5434, 4544, 2113, 3525, 5575, 5193, 2639, 5584, 856, 5213, 4708, 3303, 616, 2280, 5234, 3577}, {532}, {1264}, {4225}, {5, 4359, 2317, 790, 539, 4257, 4775, 4012, 5293, 813, 3374, 817, 6079, 578, 3782, 3143, 1235, 5338, 3297, 4076, 1390, 4721, 3830, 3838}, {3282}, {505}, {3255}, {4080}, {1728, 4827, 1077, 2502}, {1643}, {3121}, {1771}, {3554}, {1520}, {3668}, {4805}, {1336}, {656}, {5614}, {927}, {2141}, {2429}, {4391}, {2727}, {3196}, {4792}, {185}, {5519}, {764}, {5912}, {5390}, {1245}, {734}, {4595}, {4289}, {4883}, {2002}, {4403}, {2889}, {1938}, {11}, {891}, {5308}]), (0.1260727871165414, [{5390}, {532}, {4099, 4739, 1542, 3463, 20, 916, 3734, 4632, 1048, 3612, 418, 5155, 677, 2866, 4530, 1718, 5687, 4932, 2382, 6096, 4817, 4050, 5973, 6105, 5082, 1883, 3292, 605, 474, 95, 6109, 4472, 357, 3818, 2296, 4473, 5755, 638, 639}, {764}, {3458, 2444, 4621, 3994, 4765, 29, 5414, 2603, 3760, 3634, 4914, 2613, 54, 1847, 953, 5434, 4544, 2113, 3525, 5575, 5193, 2639, 5584, 856, 5213, 4708, 3303, 616, 2280, 5234, 3577}, {3554}, {4805}, {5, 4359, 2317, 790, 539, 4257, 4775, 4012, 5293, 813, 3374, 817, 6079, 578, 3782, 3143, 1235, 5338, 3297, 4076, 1390, 4721, 3830, 3838}, {4289}, {1938}, {505}, {2889}, {4451}, {4792}, {891}, {185}, {1728, 4827, 1077, 2502}, {3196}, {5308}, {5912}, {4403}, {2141}, {4225}, {3282}, {5614}, {656}, {927}, {1771}, {3668}, {1336}, {734}, {1245}, {4883}, {4595}, {2429}, {4391}, {11}, {2727}, {1264}, {3121}, {1643}, {2002}, {1520}, {3255}, {1876}, {5519}, {4080}]), (0.11767574973423461, [{3458, 2444, 4621, 3994, 4765, 29, 5414, 2603, 3760, 3634, 4914, 2613, 54, 1847, 953, 5434, 4544, 2113, 3525, 5575, 5193, 2639, 5584, 856, 5213, 4708, 3303, 616, 2280, 5234, 3577}, {4099, 4739, 1542, 3463, 20, 916, 3734, 4632, 1048, 3612, 418, 5155, 677, 2866, 4530, 1718, 5687, 4932, 2382, 6096, 4817, 4050, 5973, 6105, 5082, 1883, 3292, 605, 474, 95, 6109, 357, 3818, 4472, 4473, 5755, 638, 639}, {5, 4359, 2317, 790, 539, 4257, 4775, 4012, 5293, 813, 3374, 817, 6079, 578, 3782, 3143, 1235, 5338, 3297, 4076, 1390, 4721, 3830, 3838}, {2002}, {927}, {505}, {1643}, {5912}, {5390}, {4805}, {1245}, {1728, 4827, 1077, 2502}, {2141}, {1771}, {2889}, {1336}, {1264}, {3255}, {1520}, {4289}, {656}, {11}, {4792}, {3282}, {2296}, {5614}, {4391}, {4595}, {4225}, {3668}, {4080}, {3554}, {3196}, {764}, {2727}, {185}, {734}, {1876}, {2429}, {3121}, {532}, {1938}, {5519}, {4451}, {891}, {4403}, {5308}, {4883}]), (0.11580919871232404, [{4621}, {5, 4359, 2317, 790, 539, 4257, 4775, 4012, 5293, 813, 3374, 817, 6079, 578, 3782, 3143, 1235, 5338, 3297, 4076, 1390, 4721, 3830, 3838}, {3458, 2444, 3994, 4765, 29, 5414, 2603, 3760, 3634, 4914, 2613, 54, 1847, 953, 5434, 4544, 2113, 3525, 5575, 5193, 2639, 5584, 856, 5213, 4708, 3303, 616, 2280, 5234, 3577}, {5912}, {3121}, {185}, {4099, 4739, 1542, 3463, 20, 916, 3734, 4632, 1048, 3612, 418, 5155, 677, 2866, 4530, 1718, 5687, 4932, 2382, 6096, 4817, 4050, 5973, 6105, 5082, 1883, 3292, 605, 474, 95, 6109, 357, 3818, 4472, 4473, 5755, 638, 639}, {4451}, {1264}, {2429}, {3282}, {4225}, {2889}, {4805}, {4403}, {3255}, {4883}, {927}, {2727}, {1771}, {532}, {3668}, {1938}, {656}, {1728, 4827, 1077, 2502}, {734}, {5308}, {5614}, {4080}, {2141}, {3196}, {3554}, {2002}, {4792}, {1643}, {4391}, {4595}, {505}, {5390}, {4289}, {1245}, {1520}, {1336}, {5519}, {891}, {1876}, {11}, {2296}, {764}]), (0.1074114635539336, [{5, 4359, 2317, 790, 539, 4257, 4775, 4012, 5293, 813, 3374, 817, 6079, 578, 3782, 3143, 1235, 5338, 3297, 4076, 1390, 4721, 3830, 3838}, {3458, 2444, 3994, 4765, 29, 5414, 2603, 3760, 3634, 4914, 2613, 54, 1847, 953, 5434, 4544, 2113, 3525, 5575, 5193, 2639, 5584, 856, 5213, 4708, 3303, 616, 2280, 5234, 3577}, {4099, 4739, 1542, 3463, 20, 916, 3734, 4632, 1048, 3612, 418, 5155, 677, 4530, 1718, 5687, 4932, 2382, 6096, 4817, 4050, 5973, 6105, 5082, 1883, 3292, 605, 474, 95, 6109, 357, 3818, 4472, 4473, 5755, 638, 639}, {1728, 4827, 1077, 2502}, {1771}, {2141}, {4805}, {734}, {764}, {2889}, {4403}, {4595}, {185}, {5614}, {4289}, {4883}, {3196}, {4621}, {5308}, {1264}, {505}, {2002}, {1938}, {4391}, {3121}, {532}, {4080}, {5912}, {4451}, {1876}, {4225}, {2296}, {1245}, {891}, {1520}, {5519}, {3668}, {2727}, {1643}, {2866}, {656}, {927}, {3255}, {3554}, {11}, {2429}, {5390}, {3282}, {4792}, {1336}]), (0.1032834202471592, [{4099, 4739, 1542, 3463, 20, 916, 3734, 4632, 1048, 3612, 418, 5155, 677, 4530, 1718, 5687, 2382, 6096, 4817, 4050, 5973, 6105, 5082, 1883, 3292, 605, 474, 95, 6109, 357, 3818, 4472, 4473, 5755, 638, 639}, {2727}, {2002}, {1728, 4827, 1077, 2502}, {5, 4359, 2317, 790, 539, 4257, 4775, 4012, 5293, 813, 3374, 817, 6079, 578, 3782, 3143, 1235, 5338, 3297, 4076, 1390, 4721, 3830, 3838}, {891}, {3458, 2444, 3994, 4765, 29, 5414, 2603, 3760, 3634, 4914, 2613, 54, 1847, 953, 5434, 4544, 2113, 3525, 5575, 5193, 2639, 5584, 856, 5213, 4708, 3303, 616, 2280, 5234, 3577}, {4403}, {734}, {656}, {2296}, {2141}, {3255}, {2866}, {4391}, {11}, {4883}, {5308}, {3196}, {505}, {4080}, {1520}, {3121}, {532}, {4289}, {2429}, {4451}, {5614}, {4932}, {1336}, {4595}, {1245}, {2889}, {3668}, {4621}, {4805}, {764}, {185}, {1876}, {3282}, {5519}, {1771}, {1264}, {4225}, {3554}, {5390}, {5912}, {1643}, {1938}, {4792}, {927}]), (0.09813173942668613, [{3458, 2444, 3994, 4765, 29, 5414, 2603, 3760, 3634, 4914, 2613, 54, 1847, 953, 5434, 4544, 2113, 3525, 5575, 5193, 2639, 5584, 856, 5213, 4708, 3303, 616, 2280, 5234, 3577}, {1728, 4827, 1077, 2502}, {4099, 4739, 1542, 3463, 20, 916, 3734, 4632, 1048, 3612, 418, 5155, 677, 4530, 1718, 5687, 2382, 6096, 4817, 4050, 5973, 6105, 5082, 1883, 3292, 605, 6109, 95, 357, 3818, 4472, 4473, 5755, 638, 639}, {4403}, {505}, {2002}, {734}, {1938}, {2727}, {5, 4359, 2317, 790, 539, 4257, 4775, 4012, 5293, 813, 3374, 817, 6079, 578, 3782, 3143, 1235, 5338, 3297, 4076, 1390, 4721, 3830, 3838}, {3554}, {4883}, {5308}, {11}, {1336}, {1876}, {4451}, {764}, {1264}, {4289}, {3668}, {3196}, {2429}, {2141}, {1245}, {2866}, {532}, {1643}, {474}, {4792}, {3121}, {4595}, {927}, {1520}, {5912}, {4621}, {891}, {2889}, {1771}, {5519}, {5390}, {4805}, {5614}, {4080}, {4225}, {4391}, {2296}, {3282}, {4932}, {3255}, {185}, {656}]), (0.09545995480504321, [{5912}, {3458, 2444, 3994, 4765, 29, 5414, 2603, 3760, 3634, 4914, 2613, 54, 1847, 953, 5434, 4544, 2113, 3525, 5575, 5193, 2639, 5584, 856, 5213, 4708, 3303, 616, 2280, 5234, 3577}, {4099, 4739, 1542, 3463, 20, 916, 3734, 4632, 1048, 3612, 418, 5155, 677, 4530, 1718, 5687, 2382, 6096, 4817, 4050, 5973, 6105, 5082, 1883, 3292, 605, 6109, 95, 357, 3818, 4472, 4473, 5755, 638, 639}, {3121}, {1643}, {1245}, {4792}, {5, 4359, 2317, 790, 539, 4257, 4775, 4012, 5293, 813, 3374, 817, 6079, 578, 3782, 3143, 1235, 5338, 3297, 4076, 1390, 4721, 3830}, {11}, {2002}, {1264}, {1520}, {3255}, {2141}, {4080}, {2866}, {1728, 4827, 1077, 2502}, {474}, {4451}, {2889}, {5308}, {656}, {4391}, {3554}, {2727}, {5614}, {1771}, {4932}, {505}, {734}, {3196}, {3668}, {4805}, {1938}, {2429}, {185}, {764}, {5519}, {927}, {532}, {4595}, {4403}, {3838}, {4883}, {1876}, {1336}, {4225}, {2296}, {4621}, {5390}, {891}, {4289}, {3282}]), (0.08999078586682469, [{4099, 4739, 1542, 3463, 20, 916, 3734, 4632, 1048, 3612, 418, 5155, 677, 4530, 1718, 5687, 2382, 6096, 4817, 4050, 5973, 6105, 5082, 1883, 3292, 605, 6109, 95, 357, 3818, 4472, 4473, 5755, 638, 639}, {3458, 2444, 3994, 4765, 29, 5414, 2603, 3760, 3634, 4914, 2613, 54, 1847, 953, 5434, 4544, 2113, 3525, 5575, 5193, 5584, 856, 5213, 4708, 3303, 616, 2280, 5234, 3577}, {11}, {4932}, {3554}, {4289}, {1520}, {5, 4359, 2317, 790, 539, 4257, 4775, 4012, 5293, 813, 3374, 817, 6079, 578, 3782, 3143, 1235, 5338, 3297, 4076, 1390, 4721, 3830}, {2429}, {2296}, {1876}, {5614}, {4805}, {1336}, {4403}, {4792}, {656}, {532}, {927}, {4391}, {4080}, {3668}, {4883}, {2866}, {3282}, {2002}, {4451}, {2727}, {4621}, {4595}, {3838}, {1728, 4827, 1077, 2502}, {5390}, {2889}, {2639}, {185}, {505}, {3255}, {474}, {5519}, {5912}, {1264}, {5308}, {1245}, {3121}, {1771}, {764}, {1643}, {891}, {4225}, {734}, {1938}, {2141}, {3196}]), (0.0872715524715405, [{3458, 2444, 3994, 4765, 29, 5414, 2603, 3760, 3634, 4914, 2613, 54, 1847, 953, 5434, 4544, 2113, 3525, 5575, 5193, 5584, 856, 5213, 4708, 3303, 616, 2280, 5234, 3577}, {4099, 4739, 1542, 3463, 20, 916, 3734, 4632, 1048, 3612, 418, 5155, 677, 4530, 1718, 5687, 2382, 6096, 4817, 4050, 5973, 6105, 5082, 1883, 3292, 605, 6109, 95, 357, 3818, 4472, 4473, 5755, 638, 639}, {2866}, {2002}, {3668}, {4883}, {2429}, {5390}, {2141}, {4805}, {1520}, {532}, {5519}, {5, 4359, 2317, 790, 539, 4775, 4012, 5293, 813, 3374, 817, 6079, 578, 3782, 3143, 1235, 5338, 3297, 4076, 1390, 4721, 3830}, {2727}, {3282}, {927}, {505}, {5912}, {474}, {2889}, {4080}, {1728, 4827, 1077, 2502}, {4792}, {764}, {3196}, {3255}, {185}, {5614}, {1643}, {1245}, {734}, {4225}, {3554}, {1938}, {1264}, {4289}, {4932}, {4595}, {4451}, {5308}, {656}, {4621}, {2296}, {4391}, {1336}, {2639}, {4257}, {3121}, {3838}, {11}, {4403}, {1771}, {1876}, {891}]), (0.08416993778279547, [{4099, 4739, 1542, 3463, 20, 916, 3734, 4632, 1048, 3612, 418, 5155, 677, 4530, 1718, 5687, 2382, 6096, 4817, 4050, 5973, 6105, 5082, 1883, 3292, 605, 6109, 95, 357, 3818, 4472, 4473, 5755, 638, 639}, {3458, 2444, 3994, 4765, 29, 5414, 2603, 3760, 3634, 4914, 2613, 54, 1847, 953, 5434, 4544, 2113, 3525, 5575, 5193, 5584, 856, 5213, 4708, 3303, 616, 2280, 5234, 3577}, {185}, {2141}, {5, 4359, 2317, 790, 539, 4775, 5293, 813, 3374, 817, 6079, 578, 3782, 3143, 1235, 5338, 3297, 4076, 1390, 4721, 3830}, {3196}, {1876}, {2889}, {2002}, {1245}, {2639}, {2866}, {4621}, {734}, {4792}, {532}, {4391}, {3121}, {4012}, {2296}, {656}, {4883}, {505}, {5614}, {1728, 4827, 1077, 2502}, {1264}, {4289}, {3668}, {1520}, {474}, {3838}, {5308}, {3255}, {4932}, {1938}, {4080}, {11}, {891}, {4595}, {1771}, {4451}, {3554}, {1336}, {1643}, {764}, {4403}, {4257}, {4225}, {5912}, {927}, {5390}, {2429}, {3282}, {4805}, {5519}, {2727}]), (0.07726055900935325, [{3458, 2444, 3994, 4765, 29, 5414, 2603, 3760, 3634, 4914, 2613, 54, 1847, 953, 5434, 4544, 2113, 3525, 5575, 5193, 5584, 856, 5213, 4708, 3303, 616, 2280, 5234, 3577}, {4739, 1542, 3463, 20, 916, 3734, 4632, 1048, 3612, 418, 5155, 677, 4530, 1718, 5687, 2382, 6096, 4817, 4050, 5973, 6105, 5082, 1883, 3292, 605, 6109, 95, 357, 3818, 4472, 4473, 5755, 638, 639}, {5, 4359, 2317, 790, 539, 4775, 5293, 813, 3374, 817, 6079, 578, 3782, 3143, 1235, 5338, 3297, 4076, 1390, 4721, 3830}, {4805}, {474}, {734}, {4792}, {4391}, {4012}, {5390}, {1728, 4827, 1077, 2502}, {185}, {4883}, {532}, {2866}, {3282}, {1876}, {4225}, {505}, {891}, {11}, {4257}, {3838}, {5308}, {2141}, {4621}, {1520}, {3196}, {4451}, {2639}, {4080}, {1771}, {3668}, {5912}, {2727}, {3121}, {927}, {1245}, {5519}, {4099}, {2002}, {1336}, {4595}, {5614}, {2889}, {3255}, {4932}, {764}, {2429}, {2296}, {4289}, {4403}, {3554}, {1938}, {1643}, {656}, {1264}]), (0.07145785310348124, [{1728, 4827, 1077, 2502}, {5, 4359, 2317, 790, 539, 4775, 5293, 813, 3374, 817, 6079, 578, 3782, 3143, 1235, 5338, 3297, 4076, 1390, 4721, 3830}, {4792}, {4739, 1542, 20, 916, 3734, 4632, 1048, 3612, 418, 5155, 677, 4530, 1718, 5687, 2382, 6096, 4817, 4050, 5973, 6105, 5082, 1883, 3292, 605, 6109, 95, 357, 3818, 4472, 4473, 5755, 638, 639}, {2002}, {3458, 2444, 3994, 4765, 29, 5414, 2603, 3760, 3634, 4914, 2613, 54, 1847, 953, 5434, 4544, 2113, 3525, 5575, 5193, 5584, 856, 5213, 4708, 3303, 616, 2280, 5234, 3577}, {2296}, {4932}, {5308}, {4595}, {3554}, {1643}, {4621}, {3463}, {2141}, {3668}, {5390}, {5519}, {4225}, {1771}, {3255}, {4289}, {656}, {5614}, {1938}, {1245}, {2889}, {2639}, {4403}, {11}, {4080}, {764}, {4883}, {4805}, {1264}, {1520}, {4451}, {185}, {4257}, {532}, {1876}, {2429}, {3282}, {3838}, {1336}, {927}, {2727}, {5912}, {4099}, {4391}, {891}, {4012}, {3121}, {2866}, {734}, {505}, {3196}, {474}]), (0.06777359538546707, [{4739, 1542, 20, 916, 3734, 4632, 1048, 3612, 418, 5155, 677, 4530, 1718, 5687, 2382, 6096, 4817, 4050, 5973, 6105, 5082, 1883, 605, 6109, 95, 357, 3818, 4472, 4473, 5755, 638, 639}, {5, 4359, 2317, 790, 539, 4775, 5293, 813, 3374, 817, 6079, 578, 3782, 3143, 1235, 5338, 3297, 4076, 1390, 4721, 3830}, {4451}, {3121}, {3668}, {3838}, {2141}, {3458, 2444, 3994, 4765, 29, 5414, 2603, 3760, 3634, 4914, 2613, 54, 1847, 953, 5434, 4544, 2113, 3525, 5575, 5193, 5584, 856, 5213, 4708, 3303, 616, 2280, 5234, 3577}, {4805}, {1938}, {3292}, {3282}, {891}, {3554}, {1728, 4827, 1077, 2502}, {764}, {4257}, {1264}, {1520}, {2866}, {185}, {1771}, {2727}, {3463}, {1245}, {4883}, {5912}, {4621}, {4080}, {2296}, {3196}, {4099}, {5308}, {4391}, {1643}, {1876}, {2429}, {532}, {474}, {505}, {4595}, {11}, {5390}, {4792}, {2889}, {3255}, {5519}, {2002}, {4289}, {4012}, {5614}, {2639}, {927}, {734}, {4225}, {1336}, {656}, {4932}, {4403}]), (0.06901493903704775, [{4739, 1542, 20, 916, 3734, 4632, 1048, 3612, 418, 5155, 677, 4530, 1718, 5687, 2382, 6096, 4817, 4050, 5973, 6105, 5082, 1883, 605, 6109, 95, 357, 3818, 4472, 4473, 5755, 638, 639}, {5, 4359, 2317, 790, 539, 4775, 5293, 813, 3374, 817, 6079, 578, 3782, 3143, 1235, 5338, 3297, 4076, 1390, 4721, 3830}, {3458, 2444, 3994, 4765, 29, 2603, 3760, 3634, 4914, 2613, 54, 1847, 953, 5434, 4544, 2113, 3525, 5575, 5193, 5584, 856, 5213, 4708, 3303, 616, 2280, 5234, 3577}, {505}, {4805}, {5519}, {2639}, {5614}, {474}, {4792}, {4932}, {2866}, {764}, {2727}, {1771}, {1245}, {5414}, {3838}, {927}, {532}, {2002}, {3292}, {4225}, {2296}, {4621}, {2889}, {4289}, {2429}, {1728, 4827, 1077, 2502}, {4099}, {11}, {4080}, {4403}, {734}, {3121}, {4451}, {1876}, {656}, {3282}, {1938}, {1336}, {891}, {4257}, {3463}, {1643}, {4883}, {3554}, {5912}, {5390}, {1264}, {5308}, {1520}, {3668}, {4595}, {3255}, {4391}, {2141}, {185}, {3196}, {4012}]), (0.06550721866802223, [{2639}, {2429}, {3458, 2444, 3994, 4765, 29, 2603, 3760, 3634, 4914, 2613, 54, 1847, 953, 5434, 4544, 2113, 3525, 5575, 5193, 5584, 856, 5213, 4708, 3303, 616, 2280, 5234, 3577}, {4739, 1542, 20, 916, 3734, 4632, 1048, 3612, 418, 5155, 677, 4530, 1718, 5687, 2382, 6096, 4817, 4050, 5973, 6105, 5082, 1883, 605, 6109, 95, 357, 3818, 4472, 4473, 5755, 638, 639}, {5, 4359, 2317, 790, 539, 4775, 5293, 813, 3374, 817, 6079, 3782, 3143, 1235, 5338, 3297, 4076, 1390, 4721, 3830}, {505}, {4225}, {764}, {3463}, {2889}, {4883}, {4621}, {4792}, {3554}, {1520}, {656}, {1771}, {185}, {4595}, {1728, 4827, 1077, 2502}, {2141}, {474}, {3255}, {734}, {4391}, {4080}, {2002}, {3838}, {927}, {4403}, {2866}, {1876}, {4257}, {2296}, {1643}, {891}, {1336}, {5519}, {1938}, {1264}, {3196}, {1245}, {3292}, {3282}, {5614}, {5414}, {578}, {4099}, {4805}, {5912}, {3121}, {4451}, {4932}, {5390}, {3668}, {4289}, {5308}, {532}, {4012}, {11}, {2727}]), (0.0640390977894803, [{3458, 2444, 3994, 4765, 29, 2603, 3760, 3634, 4914, 2613, 54, 1847, 953, 5434, 4544, 2113, 3525, 5575, 5193, 5584, 856, 5213, 4708, 3303, 616, 2280, 5234, 3577}, {5, 4359, 2317, 790, 539, 4775, 5293, 813, 3374, 817, 6079, 3782, 3143, 1235, 5338, 3297, 4076, 1390, 4721, 3830}, {4739, 1542, 20, 916, 4632, 1048, 3612, 418, 5155, 677, 4530, 1718, 5687, 2382, 6096, 4817, 4050, 5973, 6105, 5082, 1883, 605, 6109, 95, 357, 3818, 4472, 4473, 5755, 638, 639}, {764}, {4621}, {3121}, {4289}, {1728, 4827, 1077, 2502}, {1520}, {4099}, {3554}, {3196}, {5614}, {4595}, {2866}, {3292}, {5519}, {2141}, {4012}, {2727}, {4792}, {11}, {1245}, {5390}, {3838}, {3668}, {532}, {1643}, {656}, {1876}, {578}, {2889}, {891}, {4403}, {3255}, {1336}, {734}, {1938}, {927}, {4451}, {4257}, {505}, {4391}, {5414}, {4932}, {4883}, {4225}, {2639}, {2296}, {5912}, {2429}, {3282}, {2002}, {185}, {4080}, {474}, {4805}, {1771}, {3463}, {3734}, {1264}, {5308}]), (0.05951960210016638, [{4289}, {3458, 2444, 3994, 4765, 29, 2603, 3760, 3634, 4914, 2613, 54, 1847, 953, 5434, 4544, 2113, 3525, 5575, 5193, 5584, 856, 5213, 4708, 3303, 616, 2280, 5234, 3577}, {4739, 1542, 20, 916, 4632, 1048, 3612, 418, 5155, 677, 4530, 1718, 5687, 2382, 6096, 4817, 4050, 5973, 6105, 5082, 1883, 605, 6109, 95, 357, 3818, 4472, 4473, 5755, 638, 639}, {1876}, {656}, {4621}, {4595}, {5, 4359, 2317, 790, 539, 4775, 813, 3374, 817, 6079, 3782, 3143, 1235, 5338, 3297, 4076, 1390, 4721, 3830}, {3292}, {2296}, {4805}, {3554}, {1245}, {1728, 4827, 1077, 2502}, {3734}, {3282}, {4080}, {4099}, {2639}, {2889}, {891}, {5912}, {4883}, {4225}, {5293}, {532}, {5614}, {1264}, {927}, {3668}, {3463}, {5414}, {3255}, {5519}, {2727}, {11}, {4792}, {4257}, {1643}, {1520}, {578}, {4451}, {185}, {1771}, {1938}, {4403}, {505}, {2141}, {2866}, {3196}, {5390}, {5308}, {734}, {1336}, {4012}, {3121}, {4391}, {4932}, {2002}, {764}, {2429}, {474}, {3838}]), (0.05848549794522395, [{3668}, {3458, 2444, 3994, 4765, 29, 2603, 3760, 3634, 4914, 2613, 54, 1847, 953, 5434, 4544, 2113, 3525, 5575, 5193, 5584, 856, 5213, 4708, 3303, 616, 2280, 5234, 3577}, {4451}, {4099}, {4739, 1542, 20, 916, 4632, 1048, 3612, 418, 5155, 677, 4530, 1718, 5687, 2382, 6096, 4817, 4050, 5973, 6105, 5082, 1883, 605, 6109, 357, 3818, 4472, 4473, 5755, 638, 639}, {1245}, {3121}, {3734}, {2639}, {1643}, {3196}, {5293}, {5390}, {5, 4359, 2317, 790, 539, 4775, 813, 3374, 817, 6079, 3782, 3143, 1235, 5338, 3297, 4076, 1390, 4721, 3830}, {4932}, {1771}, {2002}, {4883}, {734}, {1728, 4827, 1077, 2502}, {578}, {4403}, {3463}, {1938}, {2141}, {3282}, {927}, {5519}, {4621}, {4289}, {4080}, {2889}, {4792}, {2296}, {4012}, {4595}, {474}, {505}, {1336}, {656}, {4257}, {4225}, {2866}, {95}, {1520}, {5414}, {3554}, {3838}, {4391}, {4805}, {5308}, {2429}, {3292}, {891}, {11}, {1264}, {185}, {764}, {5912}, {1876}, {3255}, {2727}, {532}, {5614}]), (0.055632989318094794, [{4739, 1542, 20, 916, 4632, 1048, 3612, 418, 5155, 677, 4530, 1718, 5687, 2382, 6096, 4817, 4050, 5973, 6105, 5082, 1883, 605, 6109, 357, 3818, 4472, 4473, 5755, 638, 639}, {11}, {3297, 5, 3782, 4775, 3143, 4359, 4076, 2317, 813, 3374, 1390, 4721, 817, 1235, 3830, 5338, 539, 6079}, {5390}, {5308}, {3458, 2444, 3994, 4765, 29, 2603, 3760, 3634, 4914, 2613, 54, 1847, 953, 5434, 4544, 2113, 3525, 5575, 5193, 5584, 856, 5213, 4708, 3303, 616, 2280, 5234, 3577}, {3838}, {1520}, {3734}, {2727}, {734}, {1728, 4827, 1077, 2502}, {3121}, {4883}, {4289}, {891}, {3196}, {5519}, {2429}, {2866}, {578}, {2889}, {1336}, {5614}, {4080}, {505}, {3554}, {474}, {185}, {4621}, {1245}, {1771}, {2639}, {5293}, {4099}, {3282}, {2002}, {4932}, {5414}, {95}, {1876}, {4403}, {4391}, {4257}, {3292}, {5912}, {4012}, {3668}, {2296}, {764}, {532}, {927}, {4792}, {4595}, {3255}, {4451}, {1264}, {1938}, {4805}, {656}, {2141}, {1643}, {3463}, {790}, {4225}]), (0.053474070117425276, [{5308}, {2639}, {532}, {4739, 1542, 20, 916, 4632, 1048, 3612, 418, 5155, 677, 4530, 1718, 5687, 2382, 6096, 4050, 5973, 6105, 5082, 1883, 605, 6109, 357, 3818, 4472, 4473, 5755, 638, 639}, {4451}, {4621}, {3458, 2444, 3994, 4765, 29, 2603, 3760, 3634, 4914, 2613, 54, 1847, 953, 5434, 4544, 2113, 3525, 5575, 5193, 5584, 856, 5213, 4708, 3303, 616, 2280, 5234, 3577}, {3297, 5, 3782, 4775, 3143, 4359, 4076, 2317, 813, 3374, 1390, 4721, 817, 1235, 3830, 5338, 539, 6079}, {2889}, {4883}, {474}, {4257}, {4080}, {11}, {3121}, {927}, {764}, {2429}, {578}, {2727}, {185}, {3668}, {505}, {891}, {3282}, {1771}, {1643}, {2141}, {3463}, {734}, {3255}, {4012}, {4595}, {5912}, {3734}, {1520}, {2002}, {4817}, {1336}, {5519}, {1728, 4827, 1077, 2502}, {4391}, {4932}, {5390}, {5614}, {1245}, {95}, {5293}, {2866}, {4099}, {3292}, {3196}, {3838}, {4403}, {5414}, {4289}, {4805}, {1938}, {656}, {3554}, {1264}, {4792}, {790}, {1876}, {4225}, {2296}]), (0.05073181011140354, [{3458, 2444, 3994, 4765, 29, 2603, 3760, 3634, 4914, 2613, 54, 1847, 953, 5434, 4544, 2113, 3525, 5575, 5193, 856, 5213, 4708, 3303, 616, 2280, 5234, 3577}, {1264}, {3282}, {1771}, {5414}, {3297, 5, 3782, 4775, 3143, 4359, 4076, 2317, 813, 3374, 1390, 4721, 817, 1235, 3830, 5338, 539, 6079}, {4391}, {4739, 1542, 20, 916, 4632, 1048, 3612, 418, 5155, 677, 4530, 1718, 5687, 2382, 6096, 4050, 5973, 6105, 5082, 1883, 605, 6109, 357, 3818, 4472, 4473, 5755, 638, 639}, {3838}, {1520}, {2296}, {4080}, {11}, {790}, {3196}, {1938}, {4451}, {3734}, {1876}, {95}, {4792}, {4621}, {3292}, {4403}, {4289}, {1728, 4827, 1077, 2502}, {4883}, {5390}, {4099}, {532}, {1245}, {4225}, {4932}, {3668}, {578}, {3554}, {3121}, {2889}, {2727}, {185}, {3255}, {927}, {474}, {505}, {1336}, {4595}, {4817}, {2866}, {764}, {3463}, {5614}, {5912}, {5308}, {4257}, {4012}, {5293}, {5519}, {2002}, {656}, {2429}, {1643}, {2141}, {891}, {5584}, {2639}, {734}, {4805}]), (0.04704127240864264, [{3196}, {3458, 2444, 3994, 4765, 29, 2603, 3760, 3634, 4914, 2613, 54, 1847, 953, 5434, 4544, 2113, 3525, 5575, 856, 5213, 4708, 3303, 616, 2280, 5234, 3577}, {3297, 5, 3782, 4775, 3143, 4359, 4076, 2317, 813, 3374, 1390, 4721, 817, 1235, 3830, 5338, 539, 6079}, {4739, 1542, 20, 916, 4632, 1048, 3612, 418, 5155, 677, 4530, 1718, 5687, 2382, 6096, 4050, 5973, 6105, 5082, 1883, 605, 6109, 357, 3818, 4472, 4473, 5755, 638, 639}, {11}, {1771}, {2296}, {3282}, {1728, 4827, 1077, 2502}, {4595}, {185}, {4099}, {4805}, {3734}, {3668}, {2727}, {5390}, {4403}, {4883}, {4012}, {734}, {2866}, {5912}, {3255}, {2889}, {4792}, {474}, {2141}, {4451}, {4225}, {764}, {1245}, {5414}, {1336}, {4080}, {891}, {3838}, {5519}, {3463}, {4817}, {927}, {1264}, {4932}, {1876}, {5584}, {4257}, {5193}, {578}, {4289}, {1643}, {790}, {5308}, {4621}, {1520}, {1938}, {3292}, {5614}, {505}, {656}, {532}, {4391}, {95}, {3554}, {3121}, {2639}, {5293}, {2002}, {2429}]), (0.04508959270461166, [{4739, 1542, 20, 916, 4632, 1048, 3612, 418, 5155, 677, 4530, 1718, 5687, 2382, 6096, 4050, 5973, 6105, 5082, 1883, 605, 6109, 357, 3818, 4472, 4473, 5755, 638, 639}, {3458, 2444, 3994, 4765, 29, 2603, 3760, 3634, 4914, 2613, 54, 1847, 953, 5434, 4544, 2113, 3525, 5575, 856, 5213, 4708, 3303, 616, 2280, 5234, 3577}, {5, 3782, 4775, 3143, 4359, 4076, 2317, 813, 3374, 1390, 4721, 817, 1235, 3830, 5338, 539, 6079}, {4391}, {2141}, {3282}, {1728, 4827, 1077, 2502}, {5390}, {3463}, {1264}, {734}, {4099}, {1771}, {5912}, {3297}, {1938}, {2639}, {11}, {3255}, {3838}, {5614}, {656}, {4883}, {4403}, {4289}, {1876}, {1520}, {2429}, {2889}, {927}, {1245}, {5519}, {474}, {5193}, {2866}, {3668}, {4595}, {4792}, {532}, {4225}, {3121}, {4621}, {4817}, {3292}, {4932}, {2002}, {764}, {4257}, {3554}, {891}, {3734}, {5293}, {3196}, {4012}, {2727}, {5414}, {4805}, {2296}, {95}, {1643}, {505}, {4080}, {4451}, {790}, {578}, {185}, {5308}, {1336}, {5584}]), (0.04139347279318739, [{4739, 1542, 20, 916, 4632, 1048, 3612, 418, 5155, 677, 4530, 1718, 5687, 2382, 6096, 4050, 5973, 6105, 5082, 1883, 6109, 357, 3818, 4472, 4473, 5755, 638, 639}, {605}, {1336}, {5, 3782, 4775, 3143, 4359, 4076, 2317, 813, 3374, 1390, 4721, 817, 1235, 3830, 5338, 539, 6079}, {1728, 4827, 1077, 2502}, {2429}, {3458, 2444, 3994, 4765, 29, 2603, 3760, 3634, 4914, 2613, 54, 1847, 953, 5434, 4544, 2113, 3525, 5575, 856, 5213, 4708, 3303, 616, 2280, 5234, 3577}, {4012}, {3255}, {1245}, {2296}, {5308}, {3838}, {3554}, {505}, {5614}, {2889}, {1643}, {3121}, {2141}, {4099}, {4792}, {764}, {790}, {4257}, {2727}, {4225}, {2002}, {1520}, {4621}, {1264}, {2866}, {4391}, {11}, {5390}, {3297}, {2639}, {4883}, {927}, {532}, {734}, {4403}, {474}, {4817}, {5584}, {891}, {3196}, {95}, {578}, {5414}, {3668}, {185}, {3463}, {1771}, {656}, {3292}, {4451}, {3282}, {1938}, {3734}, {1876}, {5193}, {4080}, {4595}, {4932}, {4289}, {5519}, {4805}, {5293}, {5912}]), (0.04612439463563711, [{2296}, {1520}, {4739, 1542, 20, 916, 4632, 1048, 3612, 418, 5155, 677, 4530, 1718, 5687, 2382, 6096, 4050, 5973, 6105, 5082, 1883, 6109, 3818, 4472, 4473, 5755, 638, 639}, {3282}, {4289}, {3458, 2444, 3994, 4765, 29, 2603, 3760, 3634, 4914, 2613, 54, 1847, 953, 5434, 4544, 2113, 3525, 5575, 856, 5213, 4708, 3303, 616, 2280, 5234, 3577}, {1643}, {2727}, {5, 3782, 4775, 3143, 4359, 4076, 2317, 813, 3374, 1390, 4721, 817, 1235, 3830, 5338, 539, 6079}, {2639}, {4403}, {5308}, {734}, {474}, {927}, {3668}, {4451}, {2866}, {1938}, {5519}, {1876}, {1728, 4827, 1077, 2502}, {1264}, {4817}, {4621}, {5414}, {891}, {357}, {4595}, {4932}, {4391}, {11}, {4883}, {3121}, {3292}, {1245}, {5614}, {4225}, {5193}, {790}, {2889}, {532}, {5584}, {3196}, {3255}, {4257}, {505}, {4805}, {5390}, {95}, {1771}, {3463}, {185}, {5912}, {4099}, {605}, {578}, {4080}, {2429}, {3838}, {3297}, {656}, {4012}, {3734}, {2141}, {1336}, {3554}, {2002}, {764}, {4792}, {5293}]), (0.04729665845500519, [{3121}, {1771}, {5, 3782, 4775, 3143, 4359, 4076, 2317, 813, 3374, 1390, 4721, 817, 1235, 3830, 5338, 539, 6079}, {790}, {3458, 2444, 3994, 4765, 29, 2603, 3760, 3634, 4914, 2613, 54, 1847, 953, 5434, 4544, 2113, 3525, 5575, 856, 5213, 4708, 3303, 616, 2280, 5234, 3577}, {4739, 1542, 20, 916, 4632, 1048, 3612, 418, 5155, 677, 4530, 1718, 5687, 2382, 6096, 4050, 6105, 5082, 1883, 6109, 3818, 4472, 4473, 5755, 638, 639}, {5614}, {4792}, {5519}, {5414}, {1336}, {4225}, {3196}, {4817}, {3734}, {532}, {1520}, {5973}, {4257}, {4080}, {4621}, {2727}, {357}, {2889}, {4805}, {95}, {1728, 4827, 1077, 2502}, {4012}, {1245}, {734}, {578}, {4403}, {1643}, {5308}, {5193}, {2429}, {3282}, {891}, {5390}, {605}, {2141}, {2002}, {4883}, {1264}, {4289}, {474}, {3292}, {4099}, {4451}, {5293}, {11}, {764}, {3463}, {5584}, {3297}, {4932}, {3554}, {4391}, {3668}, {927}, {656}, {2866}, {3255}, {505}, {2296}, {4595}, {185}, {1876}, {3838}, {5912}, {2639}, {1938}]), (0.04183028062111832, [{4080}, {11}, {1728, 4827, 1077, 2502}, {891}, {505}, {927}, {4739, 1542, 20, 916, 4632, 1048, 3612, 418, 5155, 677, 4530, 1718, 5687, 2382, 6096, 4050, 6105, 5082, 1883, 6109, 4472, 4473, 5755, 638, 639}, {656}, {2866}, {4883}, {2429}, {474}, {5193}, {4817}, {3458, 2444, 3994, 4765, 29, 2603, 3760, 3634, 4914, 2613, 54, 1847, 953, 5434, 4544, 2113, 3525, 5575, 856, 5213, 4708, 3303, 616, 2280, 5234, 3577}, {5, 3782, 4775, 3143, 4359, 4076, 2317, 813, 3374, 1390, 4721, 817, 1235, 3830, 5338, 539, 6079}, {5973}, {4932}, {3668}, {2141}, {3196}, {605}, {3463}, {3554}, {95}, {2727}, {4289}, {1245}, {5519}, {2639}, {3838}, {5414}, {4391}, {5584}, {2296}, {3292}, {1264}, {357}, {3297}, {3282}, {5912}, {4403}, {3121}, {4012}, {734}, {1771}, {764}, {5293}, {4595}, {532}, {2002}, {2889}, {5308}, {4099}, {1643}, {790}, {4225}, {4451}, {3734}, {1520}, {3818}, {4805}, {3255}, {1876}, {578}, {4257}, {4792}, {185}, {5614}, {1336}, {5390}, {1938}, {4621}]), (0.04432343456552473, [{3458, 2444, 3994, 4765, 29, 2603, 3760, 3634, 4914, 2613, 54, 1847, 953, 5434, 4544, 2113, 3525, 5575, 856, 5213, 4708, 3303, 616, 2280, 5234, 3577}, {5, 3782, 4775, 3143, 4359, 4076, 2317, 813, 3374, 1390, 4721, 817, 1235, 3830, 5338, 539, 6079}, {1728, 4827, 1077, 2502}, {4739, 1542, 20, 916, 4632, 1048, 3612, 418, 5155, 677, 4530, 1718, 5687, 2382, 6096, 4050, 6105, 5082, 6109, 4472, 4473, 5755, 638, 639}, {2727}, {4289}, {5614}, {3282}, {891}, {4403}, {3292}, {3838}, {2639}, {4805}, {1883}, {5414}, {3734}, {4595}, {2141}, {1264}, {95}, {578}, {4225}, {2296}, {474}, {790}, {4621}, {3554}, {605}, {3818}, {1771}, {5293}, {5390}, {1643}, {4099}, {4451}, {1876}, {4932}, {1336}, {2889}, {2002}, {1245}, {4257}, {185}, {656}, {4883}, {357}, {3668}, {3463}, {5193}, {2429}, {4012}, {3196}, {5973}, {734}, {3121}, {505}, {3297}, {4391}, {5308}, {5912}, {927}, {4080}, {5519}, {4817}, {1938}, {5584}, {3255}, {532}, {11}, {2866}, {764}, {1520}, {4792}]), (0.03862330174384698, [{3458, 2444, 3994, 4765, 29, 2603, 3760, 3634, 4914, 2613, 54, 1847, 953, 5434, 4544, 2113, 3525, 5575, 856, 4708, 3303, 616, 2280, 5234, 3577}, {357}, {5308}, {2141}, {5, 3782, 4775, 3143, 4359, 4076, 2317, 813, 3374, 1390, 4721, 817, 1235, 3830, 5338, 539, 6079}, {2866}, {4621}, {3818}, {4739, 1542, 20, 916, 4632, 1048, 3612, 418, 5155, 677, 4530, 1718, 5687, 2382, 6096, 4050, 6105, 5082, 6109, 4472, 4473, 5755, 638, 639}, {1728, 4827, 1077, 2502}, {4391}, {5519}, {4099}, {5193}, {505}, {4883}, {1520}, {474}, {3734}, {3282}, {4817}, {3838}, {2727}, {185}, {5213}, {4225}, {1771}, {5293}, {4932}, {2429}, {1643}, {1876}, {578}, {4403}, {11}, {4289}, {3121}, {5390}, {1938}, {2002}, {5584}, {927}, {3554}, {3668}, {1245}, {4595}, {3297}, {4451}, {2296}, {4012}, {3463}, {4805}, {5414}, {1264}, {891}, {790}, {3196}, {4080}, {2639}, {4792}, {734}, {4257}, {3255}, {5912}, {3292}, {5614}, {532}, {2889}, {95}, {605}, {5973}, {764}, {1883}, {656}, {1336}]), (0.03446944072200292, [{5, 3782, 4775, 4359, 4076, 2317, 813, 3374, 1390, 4721, 817, 1235, 3830, 5338, 539, 6079}, {11}, {1728, 4827, 1077, 2502}, {1771}, {3196}, {95}, {4739, 1542, 20, 916, 4632, 1048, 3612, 418, 5155, 677, 4530, 1718, 5687, 2382, 6096, 4050, 6105, 5082, 6109, 4472, 4473, 5755, 638, 639}, {3282}, {2639}, {1883}, {1520}, {3458, 2444, 3994, 4765, 29, 2603, 3760, 3634, 4914, 2613, 54, 1847, 953, 5434, 4544, 2113, 3525, 5575, 856, 4708, 3303, 616, 2280, 5234, 3577}, {4621}, {3297}, {2141}, {505}, {2002}, {2866}, {3734}, {1264}, {927}, {4883}, {3292}, {4225}, {5293}, {4451}, {4595}, {1876}, {4289}, {4792}, {3143}, {4012}, {5912}, {790}, {185}, {532}, {764}, {5213}, {4932}, {1643}, {357}, {1245}, {5193}, {3255}, {734}, {2889}, {4805}, {5973}, {1336}, {5519}, {605}, {4080}, {3554}, {3838}, {2296}, {3818}, {4391}, {4257}, {5390}, {5414}, {2429}, {4099}, {4403}, {5584}, {3668}, {474}, {578}, {4817}, {2727}, {5308}, {656}, {891}, {1938}, {3121}, {3463}, {5614}]), (0.03042094388868516, [{4595}, {3458, 2444, 3994, 4765, 29, 2603, 3760, 3634, 4914, 2613, 54, 1847, 953, 5434, 4544, 2113, 3525, 5575, 856, 4708, 3303, 616, 2280, 5234, 3577}, {790}, {4739, 20, 916, 4632, 1048, 3612, 418, 5155, 677, 4530, 1718, 5687, 2382, 6096, 4050, 6105, 5082, 6109, 4472, 4473, 5755, 638, 639}, {5390}, {5519}, {4792}, {5, 3782, 4775, 4359, 4076, 2317, 813, 3374, 1390, 4721, 817, 1235, 3830, 5338, 539, 6079}, {3143}, {3292}, {3463}, {1520}, {5213}, {605}, {2639}, {4403}, {2866}, {5973}, {764}, {3196}, {4080}, {4621}, {578}, {1728, 4827, 1077, 2502}, {2429}, {3282}, {1336}, {4932}, {2002}, {891}, {185}, {5308}, {4257}, {474}, {1264}, {5293}, {5584}, {532}, {4225}, {5414}, {357}, {1643}, {3297}, {734}, {3838}, {4451}, {5614}, {2889}, {4012}, {2296}, {1876}, {4099}, {505}, {4289}, {95}, {1771}, {1542}, {1245}, {1883}, {4805}, {3818}, {3668}, {5912}, {3121}, {4883}, {4817}, {2141}, {1938}, {3554}, {4391}, {3734}, {11}, {5193}, {3255}, {927}, {656}, {2727}]), (0.02444658506640557, [{5293}, {1771}, {3458, 2444, 3994, 4765, 29, 2603, 3760, 3634, 4914, 2613, 54, 1847, 5434, 4544, 2113, 3525, 5575, 856, 4708, 3303, 616, 2280, 5234, 3577}, {1728, 4827, 1077, 2502}, {4805}, {4932}, {4739, 20, 916, 4632, 1048, 3612, 418, 5155, 677, 4530, 1718, 5687, 2382, 6096, 4050, 6105, 5082, 6109, 4472, 4473, 5755, 638, 639}, {1542}, {5, 3782, 4775, 4359, 4076, 2317, 813, 3374, 1390, 4721, 817, 1235, 3830, 5338, 539, 6079}, {3734}, {578}, {2002}, {3121}, {505}, {4817}, {927}, {2296}, {4792}, {3297}, {4883}, {1643}, {4080}, {5584}, {2429}, {5308}, {3292}, {1336}, {1264}, {790}, {1938}, {5519}, {764}, {734}, {11}, {5912}, {5414}, {5973}, {185}, {1883}, {4012}, {656}, {3818}, {4225}, {5390}, {4595}, {2141}, {4099}, {3554}, {3668}, {5614}, {4289}, {1245}, {1520}, {3463}, {4403}, {3143}, {3255}, {2639}, {1876}, {5193}, {3838}, {953}, {2889}, {605}, {2866}, {4451}, {532}, {5213}, {2727}, {4391}, {3196}, {474}, {4621}, {891}, {4257}, {95}, {3282}, {357}]), (0.02384300875464751, [{3734}, {1520}, {4739, 20, 916, 4632, 1048, 3612, 418, 5155, 677, 4530, 1718, 5687, 2382, 6096, 4050, 5082, 6109, 4472, 4473, 5755, 638, 639}, {764}, {3458, 2444, 3994, 4765, 29, 2603, 3760, 3634, 4914, 2613, 54, 1847, 5434, 4544, 2113, 3525, 5575, 856, 4708, 3303, 616, 2280, 5234, 3577}, {4099}, {953}, {2727}, {2639}, {5, 3782, 4775, 4359, 4076, 2317, 813, 3374, 1390, 4721, 817, 1235, 3830, 5338, 539, 6079}, {3292}, {3818}, {474}, {1876}, {1883}, {4225}, {790}, {4792}, {3554}, {5912}, {5308}, {4403}, {3282}, {1771}, {3668}, {1336}, {5414}, {6105}, {2002}, {656}, {4289}, {5193}, {3121}, {4391}, {185}, {1938}, {5584}, {505}, {4257}, {927}, {95}, {891}, {1728, 4827, 1077, 2502}, {5614}, {605}, {3838}, {5213}, {2429}, {734}, {3196}, {3255}, {4932}, {4080}, {2889}, {3297}, {1264}, {5973}, {4595}, {4621}, {5519}, {4883}, {357}, {5390}, {1245}, {1542}, {4817}, {4805}, {4451}, {3463}, {2866}, {11}, {1643}, {4012}, {5293}, {2296}, {3143}, {578}, {532}, {2141}]), (0.021656876286742677, [{3458, 2444, 3994, 4765, 29, 2603, 3760, 3634, 4914, 2613, 54, 1847, 5434, 4544, 2113, 3525, 5575, 856, 4708, 3303, 616, 2280, 5234, 3577}, {656}, {4739, 20, 916, 1048, 3612, 418, 5155, 677, 4530, 1718, 5687, 2382, 6096, 4050, 5082, 6109, 4472, 4473, 5755, 638, 639}, {2889}, {4805}, {2639}, {3196}, {3668}, {2429}, {5, 3782, 4775, 4359, 4076, 2317, 813, 3374, 1390, 4721, 817, 1235, 3830, 5338, 539, 6079}, {5193}, {11}, {3121}, {3554}, {2296}, {1938}, {1520}, {2866}, {1876}, {3143}, {734}, {605}, {357}, {927}, {4451}, {5614}, {3463}, {4621}, {5390}, {95}, {505}, {3282}, {3292}, {578}, {1728, 4827, 1077, 2502}, {1771}, {4012}, {185}, {1883}, {4099}, {4080}, {4792}, {4632}, {532}, {764}, {474}, {4289}, {5584}, {5293}, {790}, {4883}, {5912}, {5519}, {5308}, {4817}, {1542}, {953}, {3838}, {4595}, {3255}, {2141}, {5414}, {4391}, {4225}, {1336}, {3734}, {2727}, {1245}, {2002}, {1643}, {6105}, {3297}, {5213}, {4257}, {3818}, {5973}, {4932}, {1264}, {4403}, {891}]), (0.015277807336348037, [{4739, 20, 916, 1048, 3612, 418, 5155, 677, 4530, 1718, 5687, 2382, 6096, 4050, 5082, 6109, 4472, 4473, 5755, 638, 639}, {2296}, {95}, {3458, 2444, 3994, 4765, 29, 2603, 3760, 3634, 4914, 54, 1847, 5434, 4544, 2113, 3525, 5575, 856, 4708, 3303, 616, 2280, 5234, 3577}, {4883}, {5, 3782, 4775, 4359, 4076, 2317, 813, 3374, 1390, 4721, 817, 1235, 3830, 5338, 539, 6079}, {1728, 4827, 1077, 2502}, {1264}, {505}, {4012}, {4080}, {5213}, {2613}, {3297}, {3838}, {4632}, {4257}, {4932}, {790}, {3292}, {1938}, {3818}, {3196}, {1876}, {4621}, {4099}, {1771}, {4595}, {2429}, {5308}, {927}, {764}, {3463}, {1520}, {5614}, {2141}, {2727}, {1336}, {953}, {11}, {4451}, {605}, {4391}, {185}, {6105}, {1643}, {4792}, {3282}, {532}, {3734}, {5519}, {5293}, {578}, {2889}, {1883}, {3121}, {357}, {5584}, {5414}, {4289}, {5390}, {3255}, {4225}, {656}, {1542}, {891}, {4403}, {2639}, {5973}, {3554}, {474}, {2002}, {734}, {5912}, {2866}, {3668}, {3143}, {5193}, {4817}, {4805}, {1245}]), (0.012299699014286514, [{3196}, {2002}, {790}, {1771}, {5912}, {3458, 2444, 3994, 4765, 29, 2603, 3760, 3634, 4914, 54, 1847, 5434, 4544, 2113, 3525, 5575, 856, 4708, 3303, 616, 2280, 5234, 3577}, {3292}, {3143}, {5213}, {474}, {1264}, {5, 3782, 4775, 4359, 4076, 2317, 813, 3374, 1390, 4721, 817, 1235, 3830, 5338, 539, 6079}, {4739, 20, 1048, 3612, 418, 5155, 677, 4530, 1718, 5687, 2382, 6096, 4050, 5082, 6109, 4472, 4473, 5755, 638, 639}, {4257}, {3838}, {3121}, {4080}, {4817}, {891}, {95}, {2866}, {4595}, {11}, {3818}, {1728, 4827, 1077, 2502}, {4012}, {1876}, {3734}, {1643}, {5519}, {656}, {3554}, {4099}, {6105}, {4883}, {4632}, {4403}, {953}, {927}, {4225}, {734}, {1542}, {2639}, {2727}, {578}, {5293}, {764}, {1520}, {5973}, {505}, {4932}, {2141}, {1336}, {2296}, {5390}, {3297}, {3255}, {605}, {5414}, {3668}, {357}, {532}, {4792}, {4289}, {4391}, {5584}, {4621}, {4805}, {2889}, {3463}, {2429}, {5614}, {5308}, {1938}, {1245}, {916}, {4451}, {2613}, {3282}, {1883}, {185}, {5193}]), (0.01525896738210808, [{5, 3782, 4775, 4359, 4076, 2317, 813, 3374, 1390, 4721, 817, 1235, 3830, 5338, 539, 6079}, {3458, 2444, 3994, 4765, 29, 2603, 3760, 3634, 4914, 54, 1847, 5434, 4544, 2113, 3525, 5575, 856, 4708, 3303, 616, 2280, 5234, 3577}, {605}, {2002}, {4739, 20, 1048, 3612, 418, 5155, 677, 4530, 1718, 5687, 2382, 6096, 4050, 5082, 6109, 4472, 4473, 638, 639}, {5973}, {474}, {4012}, {4792}, {1728, 4827, 1077, 2502}, {2866}, {505}, {1520}, {1883}, {3463}, {916}, {656}, {953}, {3818}, {2429}, {4403}, {1938}, {5584}, {1643}, {1876}, {4632}, {1771}, {3196}, {4805}, {3282}, {5755}, {5193}, {2613}, {2727}, {4225}, {357}, {1336}, {5308}, {3143}, {2296}, {4080}, {11}, {4099}, {891}, {6105}, {5519}, {3121}, {5614}, {532}, {5213}, {734}, {4883}, {2639}, {4621}, {3554}, {1245}, {5293}, {5912}, {927}, {4932}, {3838}, {1264}, {3297}, {790}, {4595}, {4451}, {5390}, {4257}, {185}, {4817}, {3255}, {3292}, {1542}, {95}, {2141}, {2889}, {3734}, {3668}, {5414}, {4289}, {764}, {578}, {4391}]), (0.01330658990199396, [{3458, 2444, 3994, 4765, 29, 2603, 3760, 3634, 4914, 54, 1847, 5434, 4544, 2113, 3525, 5575, 856, 4708, 3303, 616, 2280, 5234, 3577}, {505}, {2889}, {4621}, {1938}, {5414}, {3463}, {605}, {6105}, {3196}, {734}, {2429}, {4739, 20, 1048, 3612, 418, 5155, 677, 4530, 1718, 5687, 2382, 6096, 4050, 5082, 6109, 4472, 4473, 638, 639}, {916}, {4012}, {2296}, {2866}, {3782, 4775, 4359, 4076, 2317, 813, 3374, 1390, 4721, 817, 1235, 3830, 5338, 539, 6079}, {4289}, {578}, {656}, {4451}, {1245}, {5755}, {1728, 4827, 1077, 2502}, {95}, {927}, {3668}, {4792}, {4403}, {3297}, {5519}, {4099}, {1336}, {357}, {2639}, {5308}, {1542}, {4225}, {1264}, {3255}, {4805}, {891}, {5193}, {2141}, {3143}, {3838}, {4257}, {5293}, {1520}, {3121}, {3292}, {532}, {2727}, {5973}, {4932}, {4595}, {764}, {4883}, {5614}, {5390}, {5213}, {790}, {3734}, {5584}, {4391}, {4632}, {1771}, {185}, {1876}, {953}, {3554}, {474}, {4080}, {5912}, {3282}, {5}, {1643}, {4817}, {2002}, {11}, {1883}, {3818}, {2613}]), (0.00971653195517902, [{4080}, {5308}, {764}, {4739, 20, 1048, 3612, 418, 5155, 677, 4530, 1718, 5687, 2382, 6096, 4050, 5082, 6109, 4472, 4473, 638, 639}, {3782, 4775, 4076, 2317, 813, 3374, 1390, 4721, 817, 1235, 3830, 5338, 539, 6079}, {5614}, {3458, 2444, 3994, 4765, 29, 2603, 3760, 3634, 4914, 54, 1847, 5434, 4544, 2113, 3525, 5575, 856, 4708, 3303, 616, 2280, 5234, 3577}, {2613}, {505}, {4792}, {3668}, {3143}, {5414}, {1883}, {5213}, {6105}, {2141}, {4012}, {4391}, {11}, {4621}, {185}, {2866}, {5973}, {4632}, {357}, {2889}, {5293}, {2727}, {3196}, {4257}, {927}, {1245}, {790}, {532}, {578}, {1938}, {5912}, {3463}, {3554}, {734}, {4932}, {1542}, {5193}, {5755}, {5584}, {1728, 4827, 1077, 2502}, {474}, {3297}, {1643}, {4289}, {2429}, {5390}, {1264}, {3734}, {916}, {95}, {3121}, {953}, {2002}, {4099}, {891}, {4225}, {3838}, {4451}, {5519}, {3818}, {3282}, {1336}, {3292}, {4883}, {656}, {2296}, {4403}, {5}, {4359}, {4595}, {1771}, {1876}, {605}, {2639}, {4805}, {1520}, {3255}, {4817}]), (0.004619277669176747, [{3994}, {4080}, {3458, 2444, 4765, 29, 2603, 3760, 3634, 4914, 54, 1847, 5434, 4544, 2113, 3525, 5575, 856, 4708, 3303, 616, 2280, 5234, 3577}, {4257}, {916}, {3292}, {2296}, {4739, 20, 1048, 3612, 418, 5155, 677, 4530, 1718, 5687, 2382, 6096, 4050, 5082, 6109, 4472, 4473, 638, 639}, {4359}, {734}, {656}, {3143}, {532}, {4012}, {1771}, {4391}, {1728, 4827, 1077, 2502}, {605}, {3782, 4775, 4076, 2317, 813, 3374, 1390, 4721, 817, 1235, 3830, 5338, 539, 6079}, {927}, {5614}, {3282}, {3463}, {953}, {357}, {4099}, {3668}, {1264}, {5}, {3818}, {4595}, {1938}, {4289}, {2002}, {1883}, {5414}, {764}, {3554}, {4817}, {578}, {1643}, {4225}, {4805}, {2429}, {1876}, {4792}, {2727}, {3838}, {4932}, {1520}, {4632}, {4451}, {2639}, {5584}, {2866}, {5390}, {4621}, {5755}, {790}, {95}, {5519}, {1245}, {3255}, {891}, {3121}, {185}, {5973}, {5308}, {1336}, {6105}, {3196}, {1542}, {2889}, {11}, {5912}, {4883}, {5193}, {2141}, {505}, {4403}, {5213}, {5293}, {3297}, {2613}, {474}, {3734}]), (0.0018637599175787833, [{3782, 4775, 4076, 2317, 813, 3374, 1390, 4721, 817, 1235, 3830, 5338, 539, 6079}, {2889}, {4739, 20, 1048, 3612, 418, 5155, 677, 4530, 1718, 5687, 2382, 6096, 4050, 5082, 6109, 4472, 4473, 638, 639}, {5755}, {3994}, {4099}, {656}, {3458, 2444, 4765, 29, 2603, 3760, 3634, 4914, 54, 1847, 5434, 4544, 3525, 5575, 856, 4708, 3303, 616, 2280, 5234, 3577}, {532}, {3121}, {1542}, {4632}, {474}, {4883}, {5}, {185}, {4257}, {2002}, {3255}, {3463}, {5519}, {3668}, {6105}, {953}, {5308}, {4012}, {5213}, {1728, 4827, 1077, 2502}, {4451}, {1264}, {927}, {505}, {2727}, {1520}, {5293}, {1771}, {95}, {2429}, {764}, {891}, {2296}, {2866}, {4359}, {4080}, {1938}, {2141}, {734}, {3196}, {1245}, {4792}, {4805}, {3297}, {3282}, {4391}, {1336}, {2639}, {4621}, {3143}, {916}, {790}, {5614}, {5584}, {5193}, {357}, {4932}, {11}, {4403}, {2113}, {3734}, {3292}, {5414}, {2613}, {605}, {4595}, {1883}, {5912}, {3838}, {3818}, {4225}, {1876}, {5390}, {3554}, {4289}, {578}, {1643}, {4817}, {5973}]), (-0.0011164417327315934, [{2889}, {3458, 2444, 4765, 29, 2603, 3760, 3634, 4914, 1847, 5434, 4544, 3525, 5575, 856, 4708, 3303, 616, 2280, 5234, 3577}, {4403}, {1245}, {2613}, {4632}, {4739, 20, 1048, 3612, 418, 5155, 677, 4530, 1718, 5687, 2382, 6096, 4050, 5082, 6109, 4472, 4473, 638, 639}, {927}, {3782, 4775, 4076, 2317, 813, 3374, 1390, 4721, 817, 1235, 3830, 5338, 539, 6079}, {3463}, {2727}, {3121}, {3143}, {54}, {3292}, {1728, 4827, 1077, 2502}, {3838}, {4012}, {6105}, {1876}, {3282}, {5755}, {1883}, {5519}, {790}, {916}, {505}, {1771}, {1336}, {5390}, {4932}, {474}, {891}, {5213}, {1643}, {2113}, {953}, {4792}, {1938}, {1520}, {4805}, {4451}, {5614}, {4621}, {2296}, {357}, {578}, {532}, {185}, {11}, {5193}, {5584}, {4817}, {3668}, {4359}, {3255}, {2141}, {5912}, {5973}, {3196}, {2002}, {4595}, {4289}, {4080}, {2429}, {5}, {95}, {3554}, {2639}, {3734}, {764}, {5414}, {5308}, {3994}, {605}, {3818}, {734}, {1264}, {4391}, {4225}, {5293}, {4257}, {656}, {4099}, {2866}, {3297}, {1542}, {4883}]), (-0.0017186224923235738, [{3782, 4775, 4076, 2317, 813, 3374, 1390, 4721, 817, 1235, 3830, 5338, 539, 6079}, {5155, 4739, 677, 4472, 2382, 6096, 4530, 4050, 20, 1718, 5687, 1048, 4473, 5082, 3612, 6109, 638, 639}, {1728, 4827, 1077, 2502}, {4225}, {1883}, {2113}, {1876}, {5614}, {5213}, {3143}, {3458, 2444, 4765, 29, 2603, 3760, 3634, 4914, 1847, 5434, 4544, 3525, 5575, 856, 4708, 3303, 616, 2280, 5234, 3577}, {5755}, {4632}, {4099}, {4932}, {4289}, {4403}, {1938}, {5193}, {2429}, {11}, {532}, {3734}, {927}, {5414}, {4805}, {1245}, {5912}, {3292}, {185}, {4257}, {3297}, {3255}, {4012}, {357}, {5584}, {2002}, {1771}, {4817}, {1520}, {5293}, {734}, {2141}, {4080}, {3554}, {3994}, {5973}, {3838}, {578}, {2889}, {916}, {505}, {3282}, {4595}, {656}, {1264}, {4391}, {764}, {2639}, {3668}, {3121}, {2727}, {4451}, {1643}, {1336}, {4883}, {6105}, {891}, {474}, {95}, {5519}, {3196}, {1542}, {3463}, {4359}, {4792}, {953}, {2866}, {3818}, {2296}, {790}, {54}, {2613}, {5390}, {5308}, {605}, {5}, {418}, {4621}]), (-0.0035028359364451533, [{3458, 2444, 4765, 29, 2603, 3760, 3634, 4914, 1847, 5434, 4544, 3525, 5575, 856, 4708, 3303, 616, 2280, 5234, 3577}, {5519}, {3782, 4775, 4076, 2317, 813, 3374, 1390, 4721, 817, 1235, 3830, 5338, 539, 6079}, {2727}, {734}, {3668}, {5155, 4739, 677, 4472, 2382, 6096, 4530, 4050, 20, 1718, 5687, 1048, 4473, 5082, 6109, 638, 639}, {5}, {3292}, {1876}, {357}, {3734}, {2613}, {2002}, {4805}, {2141}, {4632}, {3554}, {5193}, {764}, {4792}, {2639}, {3255}, {916}, {4595}, {505}, {656}, {3463}, {1728, 4827, 1077, 2502}, {6105}, {927}, {605}, {790}, {54}, {1643}, {2866}, {4817}, {5293}, {4099}, {5584}, {3143}, {1883}, {5973}, {4932}, {418}, {1245}, {3838}, {578}, {5213}, {5414}, {4289}, {2889}, {5755}, {532}, {3818}, {4621}, {11}, {5308}, {1520}, {4451}, {4012}, {953}, {3612}, {95}, {1542}, {4359}, {4257}, {2113}, {3121}, {891}, {5614}, {1938}, {5390}, {1264}, {3282}, {1336}, {3994}, {2429}, {4391}, {2296}, {3297}, {4225}, {4080}, {185}, {3196}, {4403}, {5912}, {4883}, {1771}, {474}]), (-0.003387005106674279, [{357}, {3612}, {891}, {2429}, {5155, 4739, 677, 4472, 2382, 6096, 4530, 4050, 20, 1718, 5687, 1048, 4473, 5082, 6109, 638, 639}, {3782, 4775, 4076, 2317, 813, 1390, 4721, 817, 1235, 3830, 5338, 539, 6079}, {3297}, {4632}, {3458, 2444, 4765, 29, 2603, 3760, 3634, 4914, 1847, 5434, 4544, 3525, 5575, 856, 4708, 3303, 616, 2280, 5234, 3577}, {2141}, {3994}, {4792}, {2866}, {790}, {5193}, {4883}, {474}, {1264}, {5414}, {2889}, {605}, {4391}, {734}, {4403}, {1643}, {11}, {2113}, {3374}, {4817}, {3121}, {1883}, {5614}, {5213}, {5390}, {4805}, {4621}, {5293}, {4595}, {953}, {4012}, {3734}, {916}, {4257}, {1520}, {764}, {4289}, {5}, {4225}, {5912}, {1771}, {3143}, {3292}, {4099}, {1938}, {2639}, {6105}, {3838}, {3196}, {2296}, {4359}, {532}, {4932}, {505}, {5973}, {95}, {927}, {3554}, {1542}, {2613}, {5308}, {3668}, {4080}, {656}, {3255}, {185}, {5755}, {3463}, {1245}, {2002}, {5519}, {2727}, {3818}, {1876}, {418}, {3282}, {4451}, {1728, 4827, 1077, 2502}, {578}, {5584}, {54}, {1336}]), (-0.0054154401798309215, [{1771}, {3782, 4775, 4076, 2317, 813, 1390, 4721, 817, 1235, 5338, 539, 6079}, {3458, 2444, 4765, 29, 2603, 3760, 3634, 4914, 1847, 5434, 4544, 3525, 5575, 856, 4708, 3303, 616, 2280, 5234, 3577}, {3734}, {6105}, {5414}, {357}, {1520}, {3121}, {1876}, {1643}, {5155, 4739, 677, 4472, 2382, 6096, 4530, 4050, 20, 1718, 5687, 1048, 4473, 5082, 6109, 638, 639}, {1938}, {4289}, {4391}, {54}, {1542}, {3554}, {953}, {764}, {4883}, {891}, {2429}, {3994}, {3374}, {3255}, {4932}, {1728, 4827, 1077, 2502}, {2727}, {5973}, {578}, {2113}, {2613}, {5614}, {3838}, {1336}, {4080}, {2141}, {1883}, {3668}, {927}, {5912}, {474}, {3463}, {5213}, {4451}, {1245}, {95}, {3830}, {3282}, {3612}, {2296}, {532}, {5390}, {3143}, {185}, {605}, {4403}, {5293}, {656}, {4595}, {5}, {505}, {4792}, {4632}, {4359}, {5193}, {790}, {3818}, {1264}, {3292}, {3196}, {4817}, {4805}, {4257}, {2002}, {4621}, {418}, {5584}, {4225}, {5308}, {2889}, {5755}, {2639}, {2866}, {734}, {5519}, {916}, {4099}, {3297}, {11}, {4012}]), (-0.005320542632548719, [{927}, {5155, 4739, 677, 2382, 6096, 4530, 4050, 20, 1718, 5687, 4472, 4473, 5082, 6109, 638, 639}, {2113}, {2727}, {2429}, {4792}, {1048}, {3782, 4775, 4076, 2317, 813, 1390, 4721, 817, 1235, 5338, 539, 6079}, {3458, 2444, 4765, 29, 2603, 3760, 3634, 4914, 1847, 5434, 4544, 3525, 5575, 856, 4708, 3303, 616, 2280, 5234, 3577}, {505}, {532}, {3282}, {3297}, {1336}, {4632}, {3374}, {1771}, {4391}, {185}, {4289}, {5973}, {5584}, {2613}, {3143}, {95}, {1728, 4827, 1077, 2502}, {3196}, {734}, {2639}, {1520}, {790}, {1542}, {4883}, {916}, {2296}, {1876}, {5193}, {4257}, {4621}, {1938}, {4805}, {891}, {5}, {5390}, {5213}, {4099}, {3830}, {4359}, {578}, {4403}, {1264}, {3255}, {2141}, {3612}, {1245}, {6105}, {2889}, {3838}, {5614}, {605}, {418}, {4080}, {2002}, {4451}, {5414}, {764}, {5519}, {4817}, {1883}, {953}, {3121}, {4595}, {11}, {3668}, {357}, {474}, {3994}, {3463}, {1643}, {4012}, {4932}, {3734}, {5755}, {54}, {2866}, {3292}, {3554}, {5293}, {3818}, {5308}, {4225}, {656}, {5912}]), (-0.007003578544641453, [{2141}, {3782, 4775, 4076, 2317, 813, 1390, 4721, 817, 1235, 5338, 539, 6079}, {1728, 4827, 1077, 2502}, {6105}, {3612}, {4451}, {5155, 4739, 677, 2382, 6096, 4530, 20, 1718, 5687, 4472, 4473, 5082, 6109, 638, 639}, {4225}, {3458, 2444, 4765, 29, 2603, 3760, 3634, 4914, 1847, 5434, 4544, 3525, 5575, 856, 4708, 3303, 616, 2280, 5234, 3577}, {3734}, {3143}, {3196}, {5973}, {5414}, {4403}, {4359}, {3292}, {927}, {1643}, {4817}, {3282}, {1938}, {1883}, {3554}, {95}, {4632}, {5213}, {3994}, {4932}, {3297}, {5308}, {4391}, {5193}, {5293}, {2727}, {3838}, {54}, {418}, {3255}, {790}, {5390}, {3463}, {1876}, {1336}, {4792}, {764}, {891}, {4080}, {605}, {1264}, {2296}, {4621}, {2889}, {2429}, {1245}, {3374}, {2639}, {3830}, {3121}, {4289}, {185}, {4099}, {357}, {953}, {3668}, {4805}, {3818}, {532}, {505}, {1542}, {4257}, {5}, {5584}, {578}, {4012}, {4883}, {656}, {4595}, {1771}, {5614}, {2002}, {1048}, {1520}, {11}, {4050}, {916}, {734}, {5755}, {2113}, {2613}, {5912}, {474}, {5519}, {2866}]), (-0.01323611451761536, [{916}, {3374}, {3458, 2444, 4765, 29, 2603, 3760, 3634, 4914, 1847, 5434, 4544, 3525, 856, 4708, 3303, 616, 2280, 5234, 3577}, {5155, 4739, 677, 2382, 6096, 4530, 20, 1718, 5687, 4472, 4473, 5082, 6109, 638, 639}, {5912}, {3782, 4775, 4076, 2317, 813, 1390, 4721, 817, 1235, 5338, 539, 6079}, {1048}, {95}, {4883}, {3292}, {11}, {3121}, {2727}, {3830}, {3734}, {3463}, {2141}, {4257}, {4792}, {3297}, {5308}, {2866}, {3255}, {5755}, {5575}, {3994}, {656}, {764}, {5213}, {1728, 4827, 1077, 2502}, {3143}, {54}, {2889}, {605}, {5}, {3818}, {1883}, {1336}, {5614}, {1264}, {4289}, {185}, {4403}, {4050}, {3282}, {5519}, {5973}, {2639}, {2429}, {3668}, {4012}, {418}, {5293}, {4817}, {5414}, {532}, {1520}, {505}, {1938}, {2002}, {2113}, {4595}, {4632}, {734}, {6105}, {4359}, {4932}, {790}, {5390}, {927}, {3196}, {5584}, {891}, {4099}, {4621}, {474}, {4225}, {2296}, {1245}, {2613}, {4451}, {5193}, {3612}, {578}, {1643}, {1876}, {3554}, {357}, {953}, {1771}, {4805}, {3838}, {1542}, {4391}, {4080}]), (-0.012355521100923278, [{3782, 4775, 4076, 2317, 813, 1390, 4721, 817, 1235, 5338, 539, 6079}, {5308}, {1771}, {3458, 2444, 4765, 29, 2603, 3760, 3634, 4914, 1847, 5434, 4544, 3525, 856, 4708, 3303, 616, 2280, 5234, 3577}, {4739, 677, 2382, 6096, 4530, 20, 1718, 5687, 4472, 4473, 5082, 6109, 638, 639}, {505}, {3121}, {474}, {1542}, {11}, {953}, {1643}, {5213}, {3196}, {764}, {357}, {3838}, {2002}, {1264}, {4632}, {1883}, {3830}, {790}, {4289}, {927}, {578}, {5912}, {1728, 4827, 1077, 2502}, {605}, {3282}, {891}, {4391}, {3612}, {3255}, {532}, {3292}, {3463}, {3554}, {4080}, {2296}, {2429}, {2727}, {5575}, {4792}, {54}, {4595}, {1048}, {95}, {4012}, {4883}, {734}, {1876}, {418}, {4621}, {3143}, {6105}, {5755}, {916}, {5519}, {4359}, {5973}, {5390}, {5193}, {2141}, {2639}, {4099}, {1938}, {4805}, {5293}, {3374}, {3818}, {4050}, {2866}, {1520}, {2613}, {2113}, {4817}, {4257}, {1336}, {5414}, {3994}, {4451}, {656}, {4403}, {3297}, {1245}, {5584}, {185}, {4225}, {5}, {2889}, {3734}, {3668}, {5614}, {5155}, {4932}]), (-0.017475104221580245, [{4739, 677, 2382, 6096, 4530, 20, 1718, 5687, 4472, 4473, 5082, 6109, 638, 639}, {5}, {4099}, {532}, {3782, 4775, 4076, 2317, 813, 1390, 4721, 817, 1235, 5338, 539, 6079}, {4544, 3458, 4708, 3525, 4765, 616, 2280, 2603, 2444, 3760, 3634, 5234, 4914, 1847, 856, 3577, 5434, 29}, {4932}, {4632}, {605}, {1520}, {1938}, {11}, {95}, {1336}, {790}, {1771}, {734}, {2639}, {2002}, {4817}, {4621}, {4792}, {418}, {3668}, {4403}, {656}, {3282}, {3994}, {2429}, {3830}, {2296}, {4595}, {1264}, {5390}, {927}, {54}, {2727}, {2613}, {1728, 4827, 1077, 2502}, {5213}, {3292}, {5584}, {3121}, {1643}, {4012}, {505}, {3255}, {4257}, {5519}, {4289}, {1245}, {3612}, {4225}, {357}, {2889}, {4805}, {6105}, {2866}, {3196}, {4883}, {891}, {3554}, {916}, {3303}, {1542}, {3374}, {474}, {5614}, {5912}, {5293}, {3818}, {5973}, {3297}, {185}, {1876}, {3143}, {2113}, {5308}, {4359}, {1048}, {764}, {5575}, {3463}, {5755}, {3838}, {953}, {4050}, {4451}, {2141}, {1883}, {3734}, {4080}, {4391}, {5193}, {578}, {5414}, {5155}]), (-0.02076790955705518, [{4817}, {953}, {3782, 4775, 4076, 2317, 813, 1390, 4721, 817, 1235, 5338, 539, 6079}, {95}, {2141}, {3994}, {4544, 3458, 4708, 4765, 616, 2280, 2603, 2444, 3760, 3634, 5234, 4914, 1847, 856, 3577, 5434, 29}, {927}, {3196}, {1728, 4827, 1077, 2502}, {5390}, {4739, 677, 2382, 6096, 4530, 20, 1718, 5687, 4472, 4473, 5082, 6109, 638, 639}, {185}, {578}, {4632}, {1520}, {2002}, {5293}, {3830}, {3818}, {4289}, {4225}, {790}, {4621}, {3282}, {5584}, {5155}, {6105}, {3525}, {1245}, {1938}, {1876}, {4805}, {3463}, {3255}, {5973}, {656}, {3668}, {5519}, {3143}, {3374}, {3554}, {1264}, {1771}, {5614}, {3734}, {5414}, {4595}, {532}, {4257}, {3612}, {605}, {418}, {4359}, {2613}, {5912}, {4391}, {4099}, {3303}, {1336}, {505}, {764}, {4451}, {4050}, {1542}, {5755}, {4883}, {916}, {3297}, {3838}, {1643}, {2429}, {3292}, {4080}, {1048}, {4012}, {5193}, {1883}, {2113}, {3121}, {2889}, {357}, {4403}, {5}, {891}, {54}, {5575}, {11}, {4792}, {2639}, {2296}, {2866}, {5213}, {734}, {2727}, {5308}, {4932}, {474}]), (-0.022974975307448686, [{4544, 3458, 4708, 4765, 616, 2280, 2603, 2444, 3760, 3634, 5234, 4914, 1847, 856, 3577, 5434, 29}, {578}, {5755}, {4225}, {4739, 677, 2382, 6096, 4530, 20, 1718, 5687, 4472, 4473, 5082, 6109, 638, 639}, {2613}, {3143}, {4817}, {4403}, {5193}, {605}, {1728, 4827, 1077, 2502}, {4451}, {3303}, {953}, {891}, {927}, {5973}, {5}, {1245}, {4883}, {4932}, {1643}, {2889}, {3838}, {4595}, {3782, 4775, 2317, 813, 1390, 4721, 817, 1235, 5338, 539, 6079}, {3292}, {790}, {1264}, {1771}, {95}, {2639}, {3668}, {418}, {3297}, {1542}, {4391}, {734}, {3463}, {5155}, {2141}, {4080}, {5213}, {1520}, {3830}, {532}, {2002}, {4012}, {5575}, {5519}, {4076}, {4621}, {5308}, {4257}, {3734}, {5414}, {5912}, {1938}, {3282}, {4289}, {1883}, {916}, {3374}, {6105}, {3818}, {505}, {2429}, {54}, {3994}, {764}, {1048}, {2113}, {2296}, {185}, {1336}, {5614}, {3255}, {4099}, {4050}, {11}, {3612}, {4632}, {5293}, {2866}, {474}, {4805}, {656}, {3554}, {4792}, {5584}, {4359}, {3121}, {2727}, {5390}, {3196}, {357}, {1876}, {3525}]), (-0.023533893949897466, [{4739, 677, 2382, 6096, 4530, 20, 1718, 5687, 4472, 4473, 5082, 6109, 638, 639}, {1520}, {3463}, {5293}, {3782, 4775, 2317, 813, 1390, 4721, 1235, 5338, 539, 6079}, {3374}, {764}, {4544, 3458, 4708, 4765, 616, 2280, 2603, 2444, 3760, 3634, 5234, 4914, 1847, 856, 3577, 5434, 29}, {4621}, {3838}, {817}, {4257}, {4050}, {4805}, {3818}, {5575}, {734}, {2429}, {2727}, {1245}, {1542}, {3525}, {5519}, {2866}, {1728, 4827, 1077, 2502}, {95}, {185}, {1771}, {4451}, {1048}, {3282}, {3830}, {2002}, {4076}, {4595}, {4359}, {5}, {4632}, {5584}, {3303}, {1938}, {3292}, {357}, {605}, {418}, {4792}, {1264}, {4012}, {5614}, {2141}, {5912}, {5193}, {4883}, {656}, {2113}, {3612}, {1336}, {3143}, {2889}, {3196}, {578}, {3554}, {4403}, {5414}, {1883}, {5973}, {5155}, {3668}, {474}, {532}, {54}, {4391}, {3121}, {4932}, {5390}, {4289}, {11}, {790}, {5308}, {916}, {953}, {3297}, {3994}, {4099}, {505}, {2613}, {3255}, {891}, {5755}, {4817}, {927}, {1876}, {5213}, {4225}, {2639}, {4080}, {3734}, {2296}, {1643}, {6105}]), (-0.021277286097614034, [{5}, {3255}, {11}, {953}, {3782, 4775, 2317, 813, 1390, 4721, 1235, 5338, 539, 6079}, {4544, 3458, 4708, 4765, 616, 2280, 2603, 2444, 3760, 3634, 5234, 4914, 1847, 856, 3577, 5434, 29}, {790}, {3830}, {4739, 677, 2382, 6096, 4530, 20, 1718, 5687, 4472, 4473, 5082, 6109, 638}, {4883}, {185}, {532}, {5973}, {2727}, {3303}, {1883}, {817}, {95}, {1728, 4827, 1077, 2502}, {3297}, {656}, {2141}, {5193}, {4595}, {1264}, {4817}, {3374}, {3838}, {5575}, {1245}, {1048}, {578}, {1542}, {5293}, {3994}, {5308}, {3668}, {3734}, {3121}, {4225}, {2002}, {4805}, {4632}, {5519}, {1938}, {4080}, {3196}, {4012}, {4451}, {4403}, {4792}, {357}, {5390}, {2113}, {916}, {605}, {639}, {3818}, {927}, {4050}, {4257}, {5584}, {5755}, {3463}, {5414}, {4359}, {2889}, {5155}, {5912}, {3282}, {5213}, {4932}, {1643}, {3554}, {3525}, {2296}, {4076}, {4391}, {1336}, {2866}, {474}, {2639}, {54}, {764}, {2429}, {418}, {3292}, {4099}, {505}, {1876}, {734}, {5614}, {2613}, {4289}, {3612}, {1771}, {6105}, {891}, {4621}, {3143}, {1520}]), (-0.026516886704539555, [{5973}, {3782, 4775, 2317, 813, 1390, 4721, 1235, 5338, 539, 6079}, {605}, {1938}, {4739, 677, 2382, 6096, 4530, 20, 1718, 5687, 4472, 4473, 5082, 6109, 638}, {1048}, {2889}, {4805}, {4932}, {1771}, {4883}, {4012}, {4289}, {3297}, {5575}, {3612}, {1542}, {4544, 3458, 4708, 4765, 616, 2280, 2603, 2444, 3760, 3634, 5234, 4914, 1847, 856, 5434, 29}, {953}, {2727}, {2639}, {3292}, {3577}, {5390}, {4391}, {1728, 4827, 1077, 2502}, {4595}, {3818}, {3374}, {1520}, {3282}, {3255}, {11}, {4792}, {639}, {4050}, {5912}, {1883}, {5308}, {4225}, {2141}, {4076}, {5193}, {5755}, {578}, {891}, {4080}, {916}, {54}, {3303}, {5293}, {4403}, {1876}, {817}, {3668}, {5414}, {5519}, {3554}, {4817}, {2113}, {3525}, {4621}, {790}, {5155}, {1643}, {2429}, {357}, {5584}, {6105}, {5213}, {4359}, {5614}, {4451}, {3196}, {1336}, {1264}, {3838}, {532}, {656}, {505}, {1245}, {927}, {185}, {474}, {4099}, {2866}, {418}, {2002}, {5}, {3463}, {95}, {3121}, {4257}, {2296}, {3734}, {3143}, {734}, {3994}, {2613}, {764}, {3830}, {4632}]), (-0.029774105459783914, [{764}, {3143}, {1048}, {4739, 677, 2382, 6096, 4530, 20, 1718, 5687, 4472, 4473, 5082, 6109, 638}, {3297}, {505}, {1336}, {5755}, {4544, 3458, 4708, 4765, 616, 2280, 2603, 2444, 3760, 3634, 5234, 4914, 1847, 856, 5434, 29}, {953}, {474}, {3554}, {4289}, {3282}, {1771}, {734}, {6079, 5338, 1390, 4775}, {4621}, {5193}, {1643}, {4932}, {3196}, {4225}, {3255}, {3292}, {578}, {1235}, {4099}, {916}, {1728, 4827, 1077, 2502}, {4012}, {2429}, {790}, {3577}, {1876}, {2727}, {3818}, {2296}, {3121}, {1520}, {11}, {2317}, {1883}, {5614}, {4257}, {95}, {817}, {3830}, {4403}, {605}, {4359}, {539}, {5575}, {4391}, {2889}, {4883}, {656}, {4805}, {1542}, {2002}, {4451}, {2113}, {3374}, {2639}, {4050}, {4632}, {3668}, {813}, {4817}, {3782}, {3612}, {891}, {1245}, {3838}, {639}, {4792}, {357}, {2613}, {4080}, {4721}, {3734}, {5}, {927}, {5973}, {4076}, {3463}, {5308}, {2141}, {5519}, {418}, {3525}, {5912}, {532}, {6105}, {5584}, {3303}, {1938}, {185}, {5213}, {5390}, {4595}, {3994}, {5155}, {2866}, {54}, {1264}, {5293}, {5414}]), (-0.028556486195023573, [{4225}, {1542}, {1264}, {3282}, {6079, 5338, 1390, 4775}, {4359}, {4595}, {3303}, {4076}, {4544, 3458, 4708, 4765, 616, 2280, 2603, 2444, 3760, 5234, 4914, 1847, 856, 5434, 29}, {578}, {5155}, {5755}, {1771}, {3554}, {5}, {2141}, {5575}, {3668}, {1938}, {4739, 677, 2382, 6096, 4530, 20, 1718, 5687, 4472, 4473, 5082, 6109, 638}, {2889}, {4805}, {1728, 4827, 1077, 2502}, {2002}, {2866}, {734}, {4883}, {5614}, {5308}, {953}, {2296}, {764}, {3121}, {5912}, {3374}, {1336}, {357}, {539}, {4632}, {1643}, {4451}, {639}, {5390}, {54}, {4080}, {2613}, {5193}, {6105}, {185}, {3292}, {2113}, {11}, {95}, {2317}, {3994}, {2727}, {4721}, {4792}, {1048}, {4403}, {5213}, {3255}, {474}, {3634}, {813}, {927}, {418}, {532}, {3782}, {3196}, {5519}, {790}, {4817}, {1520}, {916}, {3830}, {4289}, {817}, {3525}, {3734}, {1883}, {1245}, {3297}, {656}, {3612}, {4391}, {2429}, {3818}, {4099}, {4257}, {3143}, {1876}, {1235}, {605}, {4050}, {4621}, {5414}, {5973}, {4932}, {3838}, {3577}, {505}, {4012}, {5293}, {2639}, {891}, {5584}, {3463}]), (-0.032681040621383536, [{656}, {2296}, {953}, {3255}, {3577}, {5519}, {3554}, {4451}, {3838}, {1938}, {4225}, {578}, {5912}, {5755}, {4739, 677, 2382, 6096, 20, 1718, 5687, 4472, 4473, 5082, 6109, 638}, {4257}, {4595}, {54}, {6079, 5338, 1390, 4775}, {3830}, {3782}, {4530}, {4544, 3458, 4708, 4765, 616, 2280, 2603, 2444, 3760, 5234, 4914, 1847, 856, 5434, 29}, {5}, {927}, {532}, {3303}, {605}, {3463}, {1048}, {3297}, {1245}, {1336}, {3818}, {3525}, {1728, 4827, 1077, 2502}, {4099}, {1643}, {2639}, {4621}, {5213}, {2141}, {1264}, {3292}, {3374}, {2002}, {3612}, {3196}, {639}, {4817}, {4359}, {3121}, {5614}, {3734}, {764}, {790}, {916}, {539}, {5414}, {4289}, {4080}, {95}, {4391}, {4050}, {3994}, {4792}, {734}, {2317}, {2113}, {505}, {185}, {3143}, {2613}, {3634}, {5293}, {3668}, {418}, {4012}, {11}, {5308}, {4883}, {1235}, {5193}, {4632}, {4076}, {4403}, {2866}, {813}, {5390}, {1771}, {5575}, {4805}, {4932}, {891}, {1876}, {3282}, {357}, {2727}, {4721}, {1883}, {817}, {5584}, {2889}, {2429}, {1520}, {5155}, {6105}, {474}, {5973}, {1542}]), (-0.03895055872675403, [{505}, {4544, 3458, 4708, 616, 2280, 2603, 2444, 3760, 5234, 1847, 856, 5434, 29}, {1048}, {639}, {357}, {4739, 677, 2382, 6096, 20, 1718, 5687, 4472, 4473, 5082, 6109, 638}, {1264}, {2866}, {3830}, {4451}, {578}, {927}, {813}, {4257}, {817}, {3612}, {4595}, {5575}, {4359}, {3143}, {2639}, {3374}, {1771}, {1728, 4827, 1077, 2502}, {6079, 5338, 1390, 4775}, {3292}, {5973}, {3838}, {4289}, {4225}, {891}, {418}, {4076}, {4099}, {3782}, {95}, {3255}, {3297}, {5308}, {4080}, {1938}, {5293}, {916}, {3121}, {54}, {1542}, {4391}, {3668}, {2727}, {2889}, {3282}, {5193}, {185}, {1520}, {1643}, {539}, {656}, {3554}, {2317}, {2296}, {5155}, {6105}, {3196}, {3734}, {4403}, {4050}, {532}, {2002}, {1883}, {605}, {4632}, {5755}, {3577}, {474}, {5}, {5912}, {4530}, {2613}, {4012}, {4765}, {1245}, {1876}, {5390}, {4805}, {5519}, {3303}, {4721}, {764}, {5614}, {11}, {5584}, {2113}, {4621}, {4792}, {734}, {5213}, {3634}, {4932}, {4914}, {3994}, {1336}, {1235}, {4817}, {2429}, {953}, {3525}, {4883}, {5414}, {790}, {2141}, {3463}, {3818}]), (-0.03485182201546342, [{532}, {418}, {5338}, {2382}, {1938}, {505}, {3303}, {4792}, {927}, {4544}, {1077}, {1847}, {3463}, {474}, {29}, {5584}, {5575}, {5155}, {3525}, {4530}, {817}, {4632}, {813}, {2296}, {2002}, {1643}, {5234}, {4359}, {1235}, {1718}, {3374}, {3760}, {95}, {3297}, {3292}, {2429}, {953}, {3734}, {2113}, {4050}, {3255}, {4473}, {4739}, {3458}, {605}, {539}, {5}, {5390}, {1728}, {4827}, {4289}, {4621}, {1245}, {1883}, {3818}, {185}, {5973}, {4257}, {6096}, {4391}, {3634}, {1048}, {1336}, {2603}, {1771}, {790}, {2889}, {4472}, {3668}, {578}, {4775}, {5614}, {5082}, {656}, {4451}, {5519}, {54}, {4883}, {5308}, {3830}, {2613}, {4932}, {11}, {4708}, {5213}, {4076}, {357}, {6079}, {2502}, {916}, {3196}, {2317}, {4012}, {1520}, {5912}, {4080}, {4765}, {4099}, {5687}, {2639}, {5434}, {639}, {3994}, {6109}, {734}, {616}, {2141}, {3838}, {891}, {5193}, {3554}, {1542}, {4817}, {4721}, {3782}, {677}, {1390}, {2727}, {856}, {638}, {6105}, {3577}, {5414}, {764}, {4595}, {2280}, {3612}, {3143}, {4805}, {3121}, {3282}, {5755}, {1264}, {5293}, {4914}, {4225}, {20}, {4403}, {1876}, {2444}, {2866}])]
rice_fb_q_data = {}
max_q_value = -100000
for i in rice_fb_q:
    components = len(i[1])
    rice_fb_q_data[components] = i[0]
    if i[0] > max_q_value:
        max_q_value = i[0]
        max_q_partition = i

plot_lines([rice_fb_q_data], "Rice Facebook", "Number of connected components", "Q value")

attributes = read_attributes("rice-facebook-undergrads.txt")

times.append(time.time())
print("Section time:")
print(len(times))
print(times[-1] - times[-2])


# Maps each ID to it's college, major, and year
max_partition_by_college = []
max_partition_by_year = []
max_partition_by_major = []
for i in max_q_partition[1]:
    college_temp = []
    year_temp = []
    major_temp = []
    for j in i:
        college_temp.append(attributes[j]['college'])
        year_temp.append(attributes[j]['year'])
        major_temp.append(attributes[j]['major'])
    max_partition_by_college.append(college_temp)
    max_partition_by_year.append(year_temp)
    max_partition_by_major.append(major_temp)

max_partition_by_college_freq = [CountFrequency(i) for i in max_partition_by_college]
max_partition_by_major_freq = [CountFrequency(i) for i in max_partition_by_major]
max_partition_by_year_freq = [CountFrequency(i) for i in max_partition_by_year]


# print(rice_fb_q)
# print(rice_fb_q_data)
# print(attributes)

plt.show()

times.append(time.time())
print("Section time:")
print(len(times))
print(times[-1] - times[-2])