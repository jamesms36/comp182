## James Sanders
## jms30
## COMP 182 Homework 5 autograder file

from typing import Tuple
from collections import *
from copy import *


def reverse_digraph_representation(graph: dict) -> dict:
    """
    Returns the given graph but in reverse direction.

    Arguments:
        graph: directed graph

    Returns:
        reversed_graph: the reverse representation of graph
    """

    # Initailizes an empty reversed graph
    reversed_graph = {}
    for i in graph.keys():
        reversed_graph[i] = {}

    # Adds all elements from graph to reversed_graph
    for i in graph.keys():
        for j in graph[i].keys():
            reversed_graph[j][i] = graph[i][j]

    return reversed_graph


def modify_edge_weights(rgraph: dict, root: int) -> None:
    """
    Modifies the weights of rgraph according to Lemma 2

    Arguments:
        rgraph: a weighted, directed graph in reversed representation
        root: the root node of rgraph

    Modifies:
        rgraph: modifies according to Lemma 2

    Returns:
        nothing
    """

    for i in rgraph.keys():
        if len(rgraph[i]) != 0 and i != root:
            min_e = min(rgraph[i].values())
            for j in rgraph[i].keys():
                # subtracts the minimum incoming weight from each edge going into a node
                rgraph[i][j] = rgraph[i][j] - min_e


def compute_rdst_candidate(rgraph: dict, root: int) -> dict:
    """
    Computes an RDST candidate on rgraph anchored at root, based on Lemma 1

    Arguments:
        rgraph: a weighted, directed graph in reversed representation
        root: the root node or rgraph
    """

    # Initializes the empty candidate graph
    candidate_rgraph = {}
    for i in rgraph.keys():
        candidate_rgraph[i] = {}

    # For every node but the root, it adds the first incoming edge that has a modified weight of zero
    for i in rgraph.keys():
        if i != root:
            for j in rgraph[i].keys():
                if rgraph[i][j] == 0:
                    candidate_rgraph[i][j] = 0
                    break

    return candidate_rgraph


def compute_cycle(rdst_candidate: dict) -> tuple:
    """
    Finds and returns a cycle within rdst_candidate, if there is one

    Arguments:
        rdst_candidate: a candidate RDST, as found by compute_rdst_candidate()

    Returns:
        a tuple of all the elements within the found cycle
    """

    for i in rdst_candidate.keys():
        j = i
        cycle = [j]

        # This starts at an arbitrary node, then goes to that node's parent node, then the parent node's parent node,
        # on an on until either it circles back to the original node, or runs in to a dead end
        while True:
            if len(rdst_candidate[j]) != 0:
                # j is set to the parent node
                j = next(iter(rdst_candidate[j]))
                if j in cycle:
                    # If it has circled back all the way to the start node
                    if j == i:
                        # Returns the cycle that was just found
                        return tuple(cycle)
                    else:
                        break
                cycle.append(j)
            else:
                break

    return tuple([])


def contract_cycle(graph: dict, cycle: tuple) -> Tuple[dict, int]:
    """
    Takes a graph and a cycle within that graph and returns a new graph where the entire cycle has
    been replaced by a single node

    Arguments:
        graph: A weighted, directed graph
        cycle: a tuple of nodes in that form a cycle within graph, where the nodes are in the cyclical order

    Returns:
        A tuple that contains a dictionary and an integer
            The dictionary is the contracted graph
            The interger is the node cstar, that has replaced the cycle
    """

    # Sets c_star to one greater than the largest node within the graph
    c_star = max(graph.keys()) + 1

    # Adds all the nodes not within the cycle to the contracted graph, then adds c_star as well
    contracted_graph = {}
    for i in graph.keys():
        if i not in cycle:
            contracted_graph[i] = {}
    contracted_graph[c_star] = {}

    # Loops over all nodes within graph
    for u in graph:
        for v in graph[u].keys():
            w = graph[u][v]

            # Case 1: if neither u nor v is in the cycle
            if u not in cycle and v not in cycle:
                # Adds all of these edges
                contracted_graph[u][v] = w

            # Case 2: if v is in the cycle and u is not
            if u not in cycle and v in cycle:
                # Adds the edge from u to c* with the smallest weight w(u,c*)
                if c_star in contracted_graph[u].keys():
                    if graph[u][v] <= contracted_graph[u][c_star]:
                        contracted_graph[u][c_star] = graph[u][v]
                else:
                    contracted_graph[u][c_star] = graph[u][v]

            # Case 3: if u is in the cycle and v is not
            if u in cycle and v not in cycle:
                # Adds the edge from c* to v with the smallest weight w(c*,v)
                if v in contracted_graph[c_star].keys():
                    if graph[u][v] <= contracted_graph[c_star][v]:
                        contracted_graph[c_star][v] = graph[u][v]
                else:
                    contracted_graph[c_star][v] = graph[u][v]

    return tuple([contracted_graph,c_star])


def expand_graph(original_graph: dict, rdst_candidate: dict, cycle: tuple, cstar: int) -> dict:
    """
    Takes an RDST candidate that has a cycle represented by cstar, and expands cstar into the full cycle

    Arguments:
        original_graph: the weighed digraph whose cycle was contracted
        rdst_candidate: a weighed digraph that was computed on the contracted version of original_graph
        cycle: a tuple of the nodes in the contracted cycle
        cstar: the node that replaced the cycle in the contracted graph

    Returns:
        expanded_graph: a weighted digraph that results from expanding the cycle in rdst_candidate
    """


    # Adds all nodes to the graph
    expanded_graph = {}
    for i in original_graph.keys():
        expanded_graph[i] = {}

    u_in = []
    v_out = []
    for i in rdst_candidate.keys():
        for j in rdst_candidate[i].keys():
            # Adds all edges not connected to the cycle
            if i != cstar and j != cstar:
                expanded_graph[i][j] = rdst_candidate[i][j]
            # Finds the nodes that connect into the cycle
            if j == cstar:
                u_in.append(i)
            # Finds the nodes that connect out of the cycle
            if i == cstar:
                v_out.append(j)


    # Adds all edges coming into the cycle
    vstar = float('inf')
    min_vstar_weight = float('inf')
    for u in u_in:
        for j in original_graph[u].keys():
            # This is the kicker right here
            if j in cycle:
                w = original_graph[u][j]
                # Makes sure to only adds the edge from u to v with minimum weight
                if w < min_vstar_weight:
                    vstar = j
                    min_vstar_weight = w
        expanded_graph[u][vstar] = min_vstar_weight


    # Adds all edges going out of the cycle
    ustar = float('inf')
    min_ustar_weight = float('inf')
    for v in v_out:
        for i in cycle:
            for j in original_graph[i].keys():
                if j == v:
                    w = original_graph[i][j]
                    # Makes sure to only adds the edge from u to v with minimum weight
                    if w < min_ustar_weight:
                        ustar = i
                        min_ustar_weight = w
        expanded_graph[ustar][v] = min_ustar_weight


    # Adds all the edges within the cycle
    for i in range(1, len(cycle)):
        if cycle[i - 1] != vstar:
            expanded_graph[cycle[i]][cycle[i - 1]] = original_graph[cycle[i]][cycle[i - 1]]
    if cycle[-1] != vstar:
        expanded_graph[cycle[0]][cycle[-1]] = original_graph[cycle[0]][cycle[-1]]

    return expanded_graph



# g = {0: {1: 20, 2: 4, 3: 20}, 1: {2: 2, 5: 16}, 2: {3: 8, 4: 20}, 3: {4: 4, 5: 8}, 4: {1: 4}, 5: {}}
# rg = reverse_digraph_representation(g)
# modify_edge_weights(rg,0)
# rg_cand = compute_rdst_candidate(rg,0)
# cycle = compute_cycle(rg_cand)
# g = reverse_digraph_representation(rg)
# (contracted_g, cstar) = contract_cycle(g, cycle)
# rdmst = expand_graph(g, contracted_g, cycle, cstar)
# rdmst_comp = compute_rdmst(g,0)

# print("Tests:")
# print("g0")
# g0 = {0: {1: 2, 2: 2, 3: 2}, 1: {2: 2, 5: 2}, 2: {3: 2, 4: 2}, 3: {4: 2, 5: 2}, 4: {1: 2}, 5: {}}
# r0 = compute_rdmst(g0, 0)
# print(r0)
# print("Expected result:")
# print("({0: {1: 2, 2: 2, 3: 2}, 1: {5: 2}, 2: {4: 2}, 3: {}, 4: {}, 5: {}}, 10)")
#
# print("")
# print("g1")
# g1 = {0: {1: 20, 2: 4, 3: 20}, 1: {2: 2, 5: 16}, 2: {3: 8, 4: 20}, 3: {4: 4, 5: 8}, 4: {1: 4}, 5: {}}
# r1 = compute_rdmst(g1, 0)
# print(r1)
# print("Expected result:")
# print("({0: {2: 4}, 1: {}, 2: {3: 8}, 3: {4: 4, 5: 8}, 4: {1: 4}, 5: {}}, 28)")
#
# print("")
# print("g2")
# g2 = {0: {1: 5, 2: 4}, 1: {2: 2}, 2: {1: 2}}
# r2 = compute_rdmst(g2, 0)
# print(r2)
# print("Expected result:")
# print("({0: {2: 4}, 1: {}, 2: {1: 2}}, 6)")
#
# print("")
# print("g3")
# g3 = {1: {2: 2.1, 3: 1.0, 4: 9.1, 5: 1.1}, 2: {1: 2.1, 3: 1.0, 4: 17.0, 5: 1.0}, 3: {1: 1.0, 2: 1.0, 4: 16.0, 5: 0.0}, 4: {1: 9.1, 2: 17.1, 3: 16.0, 5: 16.0}, 5: {1: 1.1, 2: 1.0, 3: 0.0, 4: 16.0}}
# r3 = compute_rdmst(g3, 1)
# print(r3)
# print("Expected result:")
# print("({1: {3: 1.0, 4: 9.1}, 2: {}, 3: {2: 1.0, 5: 0.0}, 4: {}, 5: {}}, 11.1)")
#
# print("")
# print("g4")
# g4 = {1: {2: 2.1, 3: 1.0, 4: 9.1, 5: 1.1, 6: 10.1, 7: 10.1, 8: 6.1, 9: 11.0, 10: 10.1}, 2: {1: 2.1, 3: 1.0, 4: 17.0, 5: 1.0, 6: 18.1, 7: 18.1, 8: 14.1, 9: 19.1, 10: 18.0}, 3: {1: 1.0, 2: 1.0, 4: 16.0, 5: 0.0, 6: 17.0, 7: 17.0, 8: 13.1, 9: 18.1, 10: 17.0}, 4: {1: 9.1, 2: 17.1, 3: 16.0, 5: 16.0, 6: 5.1, 7: 5.1, 8: 15.1, 9: 6.1, 10: 5.0}, 5: {1: 1.1, 2: 1.0, 3: 0.0, 4: 16.0, 6: 17.1, 7: 17.1, 8: 13.1, 9: 18.1, 10: 17.0}, 6: {1: 10.1, 2: 18.1, 3: 17.0, 4: 5.1, 5: 17.1, 7: 0.0, 8: 16.1, 9: 7.1, 10: 0.0}, 7: {1: 10.1, 2: 18.1, 3: 17.0, 4: 5.1, 5: 17.1, 6: 0.0, 8: 16.0, 9: 7.1, 10: 0.0}, 8: {1: 6.1, 2: 14.1, 3: 13.1, 4: 15.1, 5: 13.1, 6: 16.1, 7: 16.0, 9: 17.1, 10: 16.1}, 9: {1: 11.1, 2: 19.1, 3: 18.1, 4: 6.1, 5: 18.1, 6: 7.1, 7: 7.1, 8: 17.1, 10: 7.0}, 10: {1: 10.1, 2: 18.1, 3: 17.1, 4: 5.1, 5: 17.0, 6: 0.0, 7: 0.0, 8: 16.1, 9: 7.0}}
# r4 = compute_rdmst(g4, 1)
# print(r4)
# print("Expected result:")
# print("({1: {8: 6.1, 3: 1.0, 4: 9.1}, 2: {}, 3: {2: 1.0, 5: 0.0}, 4: {9: 6.1, 10: 5.0}, 5: {}, 6: {7: 0.0}, 7: {}, 8: {}, 9: {}, 10: {6: 0.0}}, 28.3)")
