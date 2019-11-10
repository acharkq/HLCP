import pickle
import json
from utilities.data_config import *


def cause_projection(civil=False):
    with open(projection, 'rb') as f:
        cause2index, cause_tree = pickle.load(f)
        if not civil:
            return cause2index, cause_tree
        cause2index, cause_tree = pickle.load(f)
        return cause2index, cause_tree


def get_leaves(civil=False, indexes=False):
    def traverse_for_leaves(box, tree):
        for node, branch in tree.items():
            if not branch:
                box.append(node)
            else:
                traverse_for_leaves(box, branch)

    cause2index, cause_tree = cause_projection(civil)
    box = []
    traverse_for_leaves(box, cause_tree)
    box = set(box)
    if not indexes:
        return box
    box = set([cause2index[cause] for cause in box])
    return box


def tree_print(tree, f, rank=0):
    for node, branch in tree.items():
        for _ in range(rank):
            f.write('\t')
        f.write(node + '\n')
        tree_print(branch, f, rank + 1)


def boxbox(tree, box):
    for node, branch in tree.items():
        box.append(node)
        if branch:
            boxbox(branch, box)


def run():
    cause2index, cause_tree = cause_projection(False)
    box = []
    boxbox(cause_tree, box)
    box.sort()
    for i in range(len(box) - 1):
        if box[i] == box[i + 1]:
            print(box[i])
    with open('/home/achar/critical_cause.txt', 'w') as f:
        tree_print(cause_tree, f, 0)


if __name__ == '__main__':
    cause2index, cause_tree =  cause_projection(civil=False)
    from utilities.internal_tools import tree_find_trace
    print(tree_find_trace(cause_tree, '重婚罪'))
    print(tree_find_trace(cause_tree, '民间借贷纠纷'))