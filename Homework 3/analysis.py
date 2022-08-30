## James Sanders
## jms30
## COMP 182 Homework 3 Problem 3


import matplotlib.pyplot as plt
import pylab
import types
import time
import math
import copy
import numpy
import random
from collections import *

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

def copy_graph(g):
    """
    Return a copy of the input graph, g

    Arguments:
    g -- a graph

    Returns:
    A copy of the input graph that does not share any objects.
    """
    return copy.deepcopy(g)

def remove_node(g, n):
    """
    Return a copy of the input graph, g, with the node n removed
    Arguments:
        g --- a graph
        n --- node to be removed from g
    Returns:
        A copy of the input graph without the node n
    """
    copy_g = copy_graph(g)
    for i in copy_g[n]:
        copy_g[i].remove(n)
    copy_g.pop(n)
    return copy_g


def upa(n, m):
    """
    Generate an undirected graph with n node and m edges per node
    using the preferential attachment algorithm.

    Arguments:
    n -- number of nodes
    m -- number of edges per node

    Returns:
    undirected random graph in UPAG(n, m)
    """
    g = {}
    if m <= n:
        g = make_complete_graph(m)
        for new_node in range(m, n):
            # Find <=m nodes to attach to new_node
            totdeg = float(total_degree(g))
            nodes = list(g.keys())
            probs = []
            for node in nodes:
                probs.append(len(g[node]) / totdeg)
            mult = distinct_multinomial(m, probs)

            # Add new_node and its random neighbors
            g[new_node] = set()
            for idx in mult:
                node = nodes[idx]
                g[new_node].add(node)
                g[node].add(new_node)
    return g


def erdos_renyi(n, p):
    """
    Generate a random Erdos-Renyi graph with n nodes and edge probability p.

    Arguments:
    n -- number of nodes
    p -- probability of an edge between any pair of nodes

    Returns:
    undirected random graph in G(n, p)
    """
    g = {}

    ### Add n nodes to the graph
    for node in range(n):
        g[node] = set()

    ### Iterate through each possible edge and add it with
    ### probability p.
    for u in range(n):
        for v in range(u + 1, n):
            r = random.random()
            if r < p:
                g[u].add(v)
                g[v].add(u)

    return g


def total_degree(g):
    """
    Compute total degree of the undirected graph g.

    Arguments:
    g -- undirected graph

    Returns:
    Total degree of all nodes in g
    """
    return sum(map(len, g.values()))


def make_complete_graph(num_nodes):
    """
    Returns a complete graph containing num_nodes nodes.

    The nodes of the returned graph will be 0...(num_nodes-1) if num_nodes-1 is positive.
    An empty graph will be returned in all other cases.

    Arguments:
    num_nodes -- The number of nodes in the returned graph.

    Returns:
    A complete graph in dictionary form.
    """
    result = {}


    for node_key in range(num_nodes):
        result[node_key] = set()
        for node_value in range(num_nodes):
            if node_key != node_value:
                result[node_key].add(node_value)

    return result


def distinct_multinomial(ntrials, probs):
    """
    Draw ntrials samples from a multinomial distribution given by
    probs.  Return a list of indices into probs for all distinct
    elements that were selected.  Always returns a list with between 1
    and ntrials elements.

    Arguments:
    ntrials -- number of trials
    probs   -- probability vector for the multinomial, must sum to 1

    Returns:
    A list of indices into probs for each element that was chosen one
    or more times.  If an element was chosen more than once, it will
    only appear once in the result.
    """
    ### select ntrials elements randomly
    mult = numpy.random.multinomial(ntrials, probs)

    ### turn the results into a list of indices without duplicates
    result = [i for i, v in enumerate(mult) if v > 0]
    return result

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
        for i in range(len(data)-len(labels)):
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

def compute_largest_cc_size(g: dict) -> int:
    num_nodes = len(g.keys())
    nodes = list(g.keys())
    check_nodes = {}
    for i in nodes:
        check_nodes[i] = 0
    max_cc = 0
    for i in nodes:
        if check_nodes[i] == 0:
            check_nodes[i] = 1
            size = 0
            que = deque()
            que.append(i)
            while len(que) > 0:
                j = que.popleft()
                size = size + 1
                for h in g[j]:
                    if check_nodes[h] == 0:
                        check_nodes[h] = 1
                        que.append(h)
            max_cc = max(max_cc, size)
    return max_cc


g = read_graph("rf7.repr")
num_nodes = len(g.keys())
num_edges = total_degree(g)/2
max_possible_edges = (num_nodes**2)/2

# Choice of p
# The probability that there is an edge between any two nodes can be estimated by the number of edges within the graph
# divided by the total number of possible edges within that graph.
prob_edge = num_edges/max_possible_edges
# Choice of m
# The number of edges per node is the number of edges in the graph divided by the number of nodes within the graph.
# While this calculation returns a float number, for the function, we need to express edges per node as an integer.
# Therefore, in this calculation, I rounded down edges per node to the nearest whole number.
edge_per_node = math.floor(num_edges/num_nodes)


g_erdos = erdos_renyi(num_nodes, prob_edge)
g_upa = upa(num_nodes, edge_per_node)

g_removed_random_data = {}
g_removed_targeted_data = {}
g_erdos_removed_random_data = {}
g_erdos_removed_targeted_data = {}
g_upa_removed_random_data = {}
g_upa_removed_targeted_data = {}

g_removed_random = copy_graph(g)
g_removed_targeted = copy_graph(g)
g_erdos_removed_random = copy_graph(g_erdos)
g_erdos_removed_targeted = copy_graph(g_erdos)
g_upa_removed_random = copy_graph(g_upa)
g_upa_removed_targeted = copy_graph(g_upa)

print("2234")

degree_list_original = sorted(g.items(), key=lambda g: len(g[1]), reverse=True)
degree_list_erdos = sorted(g_erdos.items(), key=lambda g_erdos: len(g_erdos[1]), reverse=True)
degree_list_upa = sorted(g_upa.items(), key=lambda g_upa: len(g_upa[1]), reverse=True)

for i in range(math.ceil(num_nodes/5)):
    print(i)
    # Original graph
    rand_node = random.choice(list(g_removed_random.keys()))
    g_removed_random = remove_node(g_removed_random, rand_node)
    g_removed_random_data[i] = compute_largest_cc_size(g_removed_random)

    g_removed_targeted = remove_node(g_removed_targeted, degree_list_original[i][0])
    g_removed_targeted_data[i] = compute_largest_cc_size(g_removed_targeted)

    # Erdos graph
    rand_erdos_node = random.choice(list(g_erdos_removed_random.keys()))
    g_erdos_removed_random = remove_node(g_erdos_removed_random, rand_erdos_node)
    g_erdos_removed_random_data[i] = compute_largest_cc_size(g_erdos_removed_random)

    g_erdos_removed_targeted = remove_node(g_erdos_removed_targeted, degree_list_erdos[i][0])
    g_erdos_removed_targeted_data[i] = compute_largest_cc_size(g_erdos_removed_targeted)


    # UPA Graph
    rand_upa_node = random.choice(list(g_upa_removed_random.keys()))
    g_upa_removed_random = remove_node(g_upa_removed_random, rand_upa_node)
    g_upa_removed_random_data[i] = compute_largest_cc_size(g_upa_removed_random)

    g_upa_removed_targeted = remove_node(g_upa_removed_targeted, degree_list_upa[i][0])
    g_upa_removed_targeted_data[i] = compute_largest_cc_size(g_upa_removed_targeted)

attack_data = [g_removed_random_data,
               g_removed_targeted_data,
               g_erdos_removed_random_data,
               g_erdos_removed_targeted_data,
               g_upa_removed_random_data,
               g_upa_removed_targeted_data]

plot_lines(attack_data,
           "Largest connected component vs number of removed nodes",
           "Number of removed nodes",
           "Largest connected component",
           ["Original Random","Original Targeted", "Erdos Random", "Erdos Targeted", "UPA Random", "UPA Targeted"])

plt.show()

