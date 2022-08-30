import itertools
import collections
import comp182
import numpy
import random
import math

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
    q = collections.deque()

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


### Use the following function to read the
### 'rice-facebook-undergrads.txt' file and turn it into an attribute
### dictionary.

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
            year    = int(fields[2])
            major   = int(fields[3])
            
             # Store student in the dictionary
            attributes[student] = {'college': college,
                                   'year': year,
                                   'major': major}
    return attributes

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
            raise Exception("Matrix dimensions do not match for matrix multiplication: %d x %d and %d x %d" % (len(A), len(A[0]), len(B), len(B[0])))

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
