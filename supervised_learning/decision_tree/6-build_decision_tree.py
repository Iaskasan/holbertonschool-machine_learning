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
        """Return the maximum depth from the current node downwards"""
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
        """Count nodes recursively below this node (optionally only leaves)"""
        total = 0
        if self.left_child:
            total += self.left_child.count_nodes_below(only_leaves)
        if self.right_child:
            total += self.right_child.count_nodes_below(only_leaves)
        if not only_leaves:
            total += 1
        return total

    def left_child_add_prefix(self, text):
        """Add formatted prefix for the left child when printing"""
        lines = [line for line in text.split("\n") if line]
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "    |  " + x + "\n"
        return new_text.rstrip("\n")

    def right_child_add_prefix(self, text):
        """Add formatted prefix for the right child when printing"""
        lines = [line for line in text.split("\n") if line]
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "       " + x + "\n"
        return new_text.rstrip("\n")

    def __str__(self):
        """Return a string representation of this node and its children"""
        if self.is_root:
            text = f"root [feature={self.feature}, threshold={self.threshold}]"
        else:
            text = "-> node [feature={}, threshold={}]".format(self.feature,
                                                               self.threshold)
        if self.left_child:
            text += "\n" + self.left_child_add_prefix(str(self.left_child))
        if self.right_child:
            text += "\n" + self.right_child_add_prefix(str(self.right_child))
        return text.rstrip("\n") + "\n"

    def get_leaves_below(self):
        """Return all leaves found below this node"""
        if self.is_leaf:
            return [str(self)]
        leaves = []
        if self.left_child:
            leaves += self.left_child.get_leaves_below()
        if self.right_child:
            leaves += self.right_child.get_leaves_below()
        return leaves

    def update_bounds_below(self):
        """Propagate lower and upper bounds from this node to its children"""
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1*np.inf}

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

    def update_indicator(self):
        """Create the indicator function for this node"""
        def is_large_enough(A):
            """Check if samples are above lower bounds"""
            if not getattr(self, "lower", None):
                return np.ones(A.shape[0], dtype=bool)
            checks = [A[:, k] > self.lower[k] for k in self.lower]
            return np.all(np.array(checks), axis=0)

        def is_small_enough(A):
            """Check if samples are below upper bounds"""
            if not getattr(self, "upper", None):
                return np.ones(A.shape[0], dtype=bool)
            checks = [A[:, k] <= self.upper[k] for k in self.upper]
            return np.all(np.array(checks), axis=0)

        self.indicator = lambda A: np.all(
            np.array([is_large_enough(A), is_small_enough(A)]),
            axis=0
        )

    def pred(self, x):
        """Recursively predict the output for a single sample"""
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):
    """Subclass of Node, construct Leaf objects"""
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Return the depth of this leaf"""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Return 1 (this leaf counts as one node)"""
        return 1

    def __str__(self):
        """Return a string representation of this leaf"""
        return (f"-> leaf [value={self.value}]")

    def get_leaves_below(self):
        """Return a list containing this leaf"""
        return [self]

    def update_bounds_below(self):
        """Leaf implementation (does nothing)"""
        pass

    def pred(self, x):
        """Return the stored value of this leaf"""
        return self.value


class Decision_Tree():
    """Class Decision Tree, framework for building and predicting"""
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
        """Return the maximum depth of the tree"""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Count nodes in the tree (all or only leaves)"""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """Return a string representation of the tree"""
        return self.root.__str__()

    def get_leaves(self):
        """Return all leaves of the tree"""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Update bounds for all nodes starting from the root"""
        self.root.update_bounds_below()

    def update_predict(self):
        """Create a vectorized prediction function for the tree"""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.sum(
            [leaf.value * leaf.indicator(A).astype(int) for leaf in leaves],
            axis=0)

    def pred(self, x):
        """Predict the output for a single sample using recursion"""
        return self.root.pred(x)
