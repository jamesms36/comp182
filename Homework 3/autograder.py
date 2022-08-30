## James Sanders
## jms30
## COMP 182 Homework 3 Problem 3

from collections import *


test_graph = {"a": set(["b", "c", "d"]), "b": set(["a", "c", "d"]), "c": set(["a", "b", "d"]), "d": set(["a", "b", "c"])}


def compute_largest_cc_size(g: dict) -> int:
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


# Tests
# grap1 = {1:set([0]), 0:set([1]), 2:set([3]), 3:set([2])}
# grap2 = {1:set([0]), 0:set([1]), 2:set([3]), 3:set([2,4]), 4:set([3])}
# grap3 = {1:set([0]), 0:set([1]), 2:set([3]), 3:set([2,4]), 4:set([3]), 5:set([])}
# grap4 = {1:set([0]), 0:set([1]), 2:set([3,4]), 3:set([2,4]), 4:set([2,3])}
# grap5 = {1:set([]), 2:set([]), 3:set([]), 10:set([]), 12:set([])}
# grap6 = {4:set([])}
# grap7 = {4:set([]), 54:set([])}
# grap8 = {1:set([2,4]), 2:set([1,4,3]), 3:set([2,4]), 4:set([1,2,3]), 5:set([6,7]), 6:set([5,7]), 7:set([5,6]), 8:set([9]), 9:set([8]), 10:set([11,12]), 11:set([10,12]), 12:set([10,11,12]), 13:set([12])}
# grap9 = {0:set([]), 1:set([2,4]), 2:set([1,4,3]), 3:set([2,4]), 4:set([1,2,3]), 5:set([6,7]), 6:set([5,7]), 7:set([5,6]), 8:set([9]), 9:set([8]), 10:set([11,12]), 11:set([10,12]), 12:set([10,11,12]), 13:set([12])}
# grap10 = {10:set([1,2,3,4,5,6,7,8,9]), 2:set([10]), 3:set([10]), 4:set([10]), 5:set([10]), 6:set([10]), 7:set([10]), 8:set([10]), 9:set([10])}
# print(compute_largest_cc_size(grap1))
# print(compute_largest_cc_size(grap2))
# print(compute_largest_cc_size(grap3))
# print(compute_largest_cc_size(grap4))
# print(compute_largest_cc_size(grap5))
# print(compute_largest_cc_size(grap6))
# print(compute_largest_cc_size(grap7))
# print(compute_largest_cc_size(grap8))
# print(compute_largest_cc_size(grap9))
# print(compute_largest_cc_size(grap10))