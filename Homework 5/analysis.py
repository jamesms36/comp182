## James Sanders
## jms30
## COMP 182 Homework 5 analysis file

from typing import Tuple
from collections import *
from copy import *
import copy


### My problem 2 functions

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


### Functions for use in Problem 2

def bfs(graph, startnode):
    """
        Perform a breadth-first search on digraph graph starting at node startnode.

        Arguments:
        graph -- directed graph
        startnode - node in graph to start the search from

        Returns:
        The distances from startnode to each node
    """
    dist = {}

    # Initialize distances
    for node in graph:
        dist[node] = float('inf')
    dist[startnode] = 0

    # Initialize search queue
    queue = deque([startnode])

    # Loop until all connected nodes have been explored
    while queue:
        node = queue.popleft()
        for nbr in graph[node]:
            if dist[nbr] == float('inf'):
                dist[nbr] = dist[node] + 1
                queue.append(nbr)
    return dist

def compute_rdmst(graph, root):
    """
        This function checks if:
        (1) root is a node in digraph graph, and
        (2) every node, other than root, is reachable from root
        If both conditions are satisfied, it calls compute_rdmst_helper
        on (graph, root).

        Since compute_rdmst_helper modifies the edge weights as it computes,
        this function reassigns the original weights to the RDMST.

        Arguments:
        graph -- a weighted digraph in standard dictionary representation.
        root -- a node id.

        Returns:
        An RDMST of graph rooted at r and its weight, if one exists;
        otherwise, nothing.
    """

    if root not in graph:
        print("The root node does not exist")
        return

    distances = bfs(graph, root)
    for node in graph:
        if distances[node] == float('inf'):
            print("The root does not reach every other node in the graph")
            return

    rdmst = compute_rdmst_helper(graph, root)

    # reassign the original edge weights to the RDMST and computes the total
    # weight of the RDMST
    rdmst_weight = 0
    for node in rdmst:
        for nbr in rdmst[node]:
            rdmst[node][nbr] = graph[node][nbr]
            rdmst_weight += rdmst[node][nbr]

    return (rdmst, rdmst_weight)

def compute_rdmst_helper(graph, root):
    """
        Computes the RDMST of a weighted digraph rooted at node root.
        It is assumed that:
        (1) root is a node in graph, and
        (2) every other node in graph is reachable from root.

        Arguments:
        graph -- a weighted digraph in standard dictionary representation.
        root -- a node in graph.

        Returns:
        An RDMST of graph rooted at root. The weights of the RDMST
        do not have to be the original weights.
        """

    # reverse the representation of graph
    rgraph = reverse_digraph_representation(graph)

    # Step 1 of the algorithm
    modify_edge_weights(rgraph, root)

    # Step 2 of the algorithm
    rdst_candidate = compute_rdst_candidate(rgraph, root)

    # compute a cycle in rdst_candidate
    cycle = compute_cycle(rdst_candidate)

    # Step 3 of the algorithm
    if not cycle:
        return reverse_digraph_representation(rdst_candidate)
    else:
        # Step 4 of the algorithm

        g_copy = deepcopy(rgraph)
        g_copy = reverse_digraph_representation(g_copy)

        # Step 4(a) of the algorithm
        (contracted_g, cstar) = contract_cycle(g_copy, cycle)
        # cstar = max(contracted_g.keys())

        # Step 4(b) of the algorithm
        new_rdst_candidate = compute_rdmst_helper(contracted_g, root)

        # Step 4(c) of the algorithm
        rdmst = expand_graph(reverse_digraph_representation(rgraph), new_rdst_candidate, cycle, cstar)

        return rdmst


### Functions for use in Problem 3

def infer_transmap(gen_data, epi_data, patient_id):
    """
        Infers a transmission map based on genetic
        and epidemiological data rooted at patient_id

        Arguments:
        gen_data -- filename with genetic data for each patient
        epi_data -- filename with epidemiological data for each patient
        patient_id -- the id of the 'patient 0'

        Returns:
        The most likely transmission map for the given scenario as the RDMST
        of a weighted, directed, complete digraph
        """

    complete_digraph = construct_complete_weighted_digraph(gen_data, epi_data)
    return compute_rdmst(complete_digraph, patient_id)

def read_patient_sequences(filename):
    """
        Turns the bacterial DNA sequences (obtained from patients) into a list containing tuples of
        (patient ID, sequence).

        Arguments:
        filename -- the input file containing the sequences

        Returns:
        A list of (patient ID, sequence) tuples.
        """
    sequences = []
    with open(filename) as f:
        line_num = 0
        for line in f:
            if len(line) > 5:
                patient_num, sequence = line.split("\t")
                sequences.append((int(patient_num), ''.join(e for e in sequence if e.isalnum())))
    return sequences

def read_patient_traces(filename):
    """
        Reads the epidemiological data file and computes the pairwise epidemiological distances between patients

        Arguments:
        filename -- the input file containing the sequences

        Returns:
        A dictionary of dictionaries where dict[i][j] is the
        epidemiological distance between i and j.
    """
    trace_data = []
    patient_ids = []
    first_line = True
    with open(filename) as f:
        for line in f:
            if first_line:
                patient_ids = line.split()
                patient_ids = list(map(int, patient_ids))
                first_line = False
            elif len(line) > 5:
                trace_data.append(line.rstrip('\n'))
    return compute_pairwise_epi_distances(trace_data, patient_ids)

def compute_pairwise_gen_distances(sequences, distance_function):
    """
        Computes the pairwise genetic distances between patients (patients' isolate genomes)

        Arguments:
        sequences -- a list of sequences that correspond with patient id's
        distance_function -- the distance function to apply to compute the weight of the
        edges in the returned graph

        Returns:
        A dictionary of dictionaries where gdist[i][j] is the
        genetic distance between i and j.
        """
    gdist = {}
    cultures = {}

    # Count the number of differences of each sequence
    for i in range(len(sequences)):
        patient_id = sequences[i][0]
        seq = sequences[i][1]
        if patient_id in cultures:
            cultures[patient_id].append(seq)
        else:
            cultures[patient_id] = [seq]
            gdist[patient_id] = {}
    # Add the minimum sequence score to the graph
    for pat1 in range(1, max(cultures.keys()) + 1):
        for pat2 in range(pat1 + 1, max(cultures.keys()) + 1):
            min_score = float("inf")
            for seq1 in cultures[pat1]:
                for seq2 in cultures[pat2]:
                    score = distance_function(seq1, seq2)
                    if score < min_score:
                        min_score = score
            gdist[pat1][pat2] = min_score
            gdist[pat2][pat1] = min_score
    return gdist


### HELPER FUNCTIONS. ###

def find_first_positives(trace_data):
    """
        Finds the first positive test date of each patient
        in the trace data.
        Arguments:
        trace_data -- a list of data pertaining to location
        and first positive test date
        Returns:
        A dictionary with patient id's as keys and first positive
        test date as values. The date numbering starts from 0 and
        the patient numbering starts from 1.
        """
    first_pos = {}
    for pat in range(len(trace_data[0])):
        first_pos[pat + 1] = None
        for date in range(len(trace_data)):
            if trace_data[date][pat].endswith(".5"):
                first_pos[pat + 1] = date
                break
    return first_pos

def compute_epi_distance(pid1, pid2, trace_data, first_pos1, first_pos2, patient_ids):
    """
        Computes the epidemiological distance between two patients.

        Arguments:
        pid1 -- the assumed donor's index in trace data
        pid2 -- the assumed recipient's index in trace data
        trace_data -- data for days of overlap and first positive cultures
        first_pos1 -- the first positive test day for pid1
        first_pos2 -- the first positive test day for pid2
        patient_ids -- an ordered list of the patient IDs given in the text file

        Returns:
        Finds the epidemiological distance from patient 1 to
        patient 2.
        """
    first_overlap = -1
    assumed_trans_date = -1
    pid1 = patient_ids.index(pid1)
    pid2 = patient_ids.index(pid2)
    # Find the first overlap of the two patients
    for day in range(len(trace_data)):
        if (trace_data[day][pid1] == trace_data[day][pid2]) & \
                (trace_data[day][pid1] != "0"):
            first_overlap = day
            break
    if (first_pos2 < first_overlap) | (first_overlap < 0):
        return len(trace_data) * 2 + 1
    # Find the assumed transmission date from patient 1 to patient 2
    for day in range(first_pos2, -1, -1):
        if (trace_data[day][pid1] == trace_data[day][pid2]) & \
                (trace_data[day][pid1] != "0"):
            assumed_trans_date = day
            break
    sc_recip = first_pos2 - assumed_trans_date

    if first_pos1 < assumed_trans_date:
        sc_donor = 0
    else:
        sc_donor = first_pos1 - assumed_trans_date
    return sc_donor + sc_recip

def compute_pairwise_epi_distances(trace_data, patient_ids):
    """
        Turns the patient trace data into a dictionary of pairwise
        epidemiological distances.

        Arguments:
        trace_data -- a list of strings with patient trace data
        patient_ids -- ordered list of patient IDs to expect

        Returns:
        A dictionary of dictionaries where edist[i][j] is the
        epidemiological distance between i and j.
        """
    edist = {}
    proc_data = []
    # Reformat the trace data
    for i in range(len(trace_data)):
        temp = trace_data[i].split()[::-1]
        proc_data.append(temp)
    # Find first positive test days and remove the indication from the data
    first_pos = find_first_positives(proc_data)
    for pid in first_pos:
        day = first_pos[pid]
        proc_data[day][pid - 1] = proc_data[day][pid - 1].replace(".5", "")
    # Find the epidemiological distance between the two patients and add it
    # to the graph
    for pid1 in patient_ids:
        edist[pid1] = {}
        for pid2 in patient_ids:
            if pid1 != pid2:
                epi_dist = compute_epi_distance(pid1, pid2, proc_data,
                                                first_pos[pid1], first_pos[pid2], patient_ids)
                edist[pid1][pid2] = epi_dist
    return edist


### My problem 3 functions

def compute_genetic_distance(seq1, seq2):
    """
    Computes the hamming distance between two genetic sequences

    Arguments:
        seq1, seq2: two strings that have the same length and both represent genetic sequences

    Return:
        the hamming distance between seq1 and seq2
    """

    dist = 0
    # Adds 1 to the Hamming distance for each index that is different between the two
    for i in range(len(seq1)):
        if seq1[i] != seq2[i]:
            dist = dist + 1
    return dist

def construct_complete_weighted_digraph(genetic_data_filename, epidemiological_data_filename):
    """
    Constructs and returns a weighed digraph of the combined genetic-epidimiological distances between the patients

    Arguments:
        genetic_data_filename: the file name of the file of genomic sequences for all the patients
        epidemiological_data_filename: the file name of the file of all epidemiological data for the patients

    Returns:
        weighted_graph: a weighed digraph of the combined genetic-epidimiological distances between the patients
    """

    # Reads in the relevant data
    gen_data = read_patient_sequences(genetic_data_filename)
    epi_graph = read_patient_traces(epidemiological_data_filename)
    gen_graph = compute_pairwise_gen_distances(gen_data, compute_genetic_distance)

    # Finds the maximum element within epi_graph
    maxE = 0
    for i in epi_graph.keys():
        for j in epi_graph[i].keys():
            if epi_graph[i][j] > maxE:
                maxE = epi_graph[i][j]

    # Initializes the weighed digraph
    weighted_graph = {}
    for i in epi_graph.keys():
        weighted_graph[i] = {}

    # For every two patients, it computes the combined genetic-epidimiological distance between them, and adds it
    # to the weighed graph
    for i in epi_graph.keys():
        for j in epi_graph[i].keys():
            # Computes disatnce according to formula 1 from the homework 5 description document
            d = gen_graph[i][j] + 999*(epi_graph[i][j]/maxE)/100000
            weighted_graph[i][j] = d

    return weighted_graph




gen_filename = "patient_sequences.txt"
epi_filename = "patient_traces.txt"

gen_data = read_patient_sequences(gen_filename)
epi_data = read_patient_traces(epi_filename)

weighted_graph = construct_complete_weighted_digraph(gen_filename, epi_filename)

rdmst = infer_transmap(gen_filename, epi_filename, 1)

print(rdmst[0])
print(rdmst[1])

