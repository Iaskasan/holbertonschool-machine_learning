#!/usr/bin/env python3
"""Decision tree Classes"""
import numpy as np


class Node:
    """Constructor for Node objects"""
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """return the max depth from the node passed as argument"""
        if self.is_leaf:
            return self.depth
        depth = []
        if self.left_child:
            depth.append(self.left_child.max_depth_below())
        if self.right_child:
            depth.append(self.right_child.max_depth_below())
        if depth:
            return max(depth)
        else:
            return self.depth

    def count_nodes_below(self, only_leaves=False):
        """recursively count total nodes below the passed node,
        can count leaf nodes only"""
        total = 0
        if self.left_child:
            total += self.left_child.count_nodes_below(only_leaves)
        if self.right_child:
            total += self.right_child.count_nodes_below(only_leaves)
        if not only_leaves:
            total += 1
        return total

    def left_child_add_prefix(self, text):
        """Add prefix for left child, avoiding extra blank lines"""
        lines = [line for line in text.split("\n") if line]
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "    |  " + x + "\n"
        return new_text.rstrip("\n")

    def right_child_add_prefix(self, text):
        """Add prefix for right child, avoiding extra blank lines"""
        lines = [line for line in text.split("\n") if line]
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "       " + x + "\n"
        return new_text.rstrip("\n")

    def __str__(self):
        """String representation of the node and its children"""
        if self.is_root:
            text = f"root [feature={self.feature}, threshold={self.threshold}]"
        else:
            text = "-> node [feature={}, threshold={}]".format(self.feature,
                                                               self.threshold)
        if self.left_child:
            text += "\n" + self.left_child_add_prefix(str(self.left_child))
        if self.right_child:
            text += "\n" + self.right_child_add_prefix(str(self.right_child))
        return text.rstrip("\n")

    def get_leaves_below(self):
        """recursively get all leaves below a certain
        node and create a formated list of leaves"""
        if self.is_leaf:
            return [str(self)]
        leaves = []
        if self.left_child:
            leaves += self.left_child.get_leaves_below()
        if self.right_child:
            leaves += self.right_child.get_leaves_below()
        return leaves

    def update_bounds_below(self):
        if self.is_root : 
            self.upper = {0:np.inf}
            self.lower = {0:-1*np.inf}

        for child in [self.left_child, self.right_child]:
            if child is None:
                continue

            child_upper = self.upper.copy()
            child_lower = self.lower.copy()

            if child is self.left_child:
                child_lower[self.feature] = self.threshold
            else:
                child_upper[self.feature] = self.threshold

            child.lower = child_lower
            child.upper = child_upper

        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()


class Leaf(Node):
    """Subclass of Node, construct Leaf objects"""
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """return the depth of the node"""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """method called when the node is a leaf"""
        return 1

    def __str__(self):
        """string representation of a leaf"""
        return (f"-> leaf [value={self.value}]")

    def get_leaves_below(self):
        """return a list of Leaf objects"""
        return [self]
    def update_bounds_below(self):
        pass


class Decision_Tree():
    """Class Decision Tree, still a framework yet"""
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """return the max depth of the tree"""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Count nodes starting from the passed node,
        can count only leaves or all nodes below"""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """String representation of the final tree"""
        return self.root.__str__()

    def get_leaves(self):
        """call the Node.get_leaves_below method for typing convenience,
        return a list of Leaves"""
        return self.root.get_leaves_below()
    def update_bounds(self):
        self.root.update_bounds_below()
