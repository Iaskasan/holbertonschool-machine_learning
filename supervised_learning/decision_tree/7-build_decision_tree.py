#!/usr/bin/env python3
"""Decision tree Classes"""
import numpy as np


class Node:
    """Constructor for Node objects"""
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Initialize a Node with optional parameters."""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """Return the maximum depth from this node downward."""
        if self.is_leaf:
            return self.depth
        depth = []
        if self.left_child:
            depth.append(self.left_child.max_depth_below())
        if self.right_child:
            depth.append(self.right_child.max_depth_below())
        return max(depth) if depth else self.depth

    def count_nodes_below(self, only_leaves=False):
        """Count nodes below this node (optionally only leaves)."""
        total = 0
        if self.left_child:
            total += self.left_child.count_nodes_below(only_leaves)
        if self.right_child:
            total += self.right_child.count_nodes_below(only_leaves)
        if not only_leaves:
            total += 1
        return total

    def left_child_add_prefix(self, text):
        """Add formatted prefix for the left child when printing."""
        lines = [line for line in text.split("\n") if line]
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "    |  " + x + "\n"
        return new_text.rstrip("\n")

    def right_child_add_prefix(self, text):
        """Add formatted prefix for the right child when printing."""
        lines = [line for line in text.split("\n") if line]
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += "       " + x + "\n"
        return new_text.rstrip("\n")

    def __str__(self):
        """Return a string representation of this node and its children."""
        if self.is_root:
            text = f"root [feature={self.feature}, threshold={self.threshold}]"
        else:
            text = f"-> node [feature={self.feature},\
            threshold={self.threshold}]"
        if self.left_child:
            text += "\n" + self.left_child_add_prefix(str(self.left_child))
        if self.right_child:
            text += "\n" + self.right_child_add_prefix(str(self.right_child))
        return text.rstrip("\n") + "\n"

    def get_leaves_below(self):
        """Return all leaves below this node."""
        if self.is_leaf:
            return [str(self)]
        leaves = []
        if self.left_child:
            leaves += self.left_child.get_leaves_below()
        if self.right_child:
            leaves += self.right_child.get_leaves_below()
        return leaves

    def update_bounds_below(self):
        """Propagate bounds from this node to its children."""
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
        """Create an indicator function for samples reaching this node."""
        def is_large_enough(A):
            """Check if samples are above lower bounds."""
            if not getattr(self, "lower", None):
                return np.ones(A.shape[0], dtype=bool)
            checks = [A[:, k] > self.lower[k] for k in self.lower]
            return np.all(np.array(checks), axis=0)

        def is_small_enough(A):
            """Check if samples are below upper bounds."""
            if not getattr(self, "upper", None):
                return np.ones(A.shape[0], dtype=bool)
            checks = [A[:, k] <= self.upper[k] for k in self.upper]
            return np.all(np.array(checks), axis=0)

        self.indicator = lambda A: np.all(
            np.array([is_large_enough(A), is_small_enough(A)]),
            axis=0
        )

    def pred(self, x):
        """Recursively predict the output for one sample."""
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        return self.right_child.pred(x)


class Leaf(Node):
    """Subclass of Node, representing a leaf."""
    def __init__(self, value, depth=None):
        """Initialize a Leaf with a value and optional depth."""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Return the depth of this leaf."""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Return 1 (this leaf is one node)."""
        return 1

    def __str__(self):
        """Return a string representation of this leaf."""
        return f"-> leaf [value={self.value}]"

    def get_leaves_below(self):
        """Return a list containing this leaf."""
        return [self]

    def update_bounds_below(self):
        """Leaf implementation of update_bounds_below (does nothing)."""
        pass

    def pred(self, x):
        """Return the stored value of this leaf."""
        return self.value


class Decision_Tree():
    """Class for building and using a decision tree."""
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """Initialize the decision tree with parameters."""
        self.rng = np.random.default_rng(seed)
        self.root = root if root else Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """Return the maximum depth of the tree."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Count nodes in the tree (all or only leaves)."""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """Return a string representation of the tree."""
        return self.root.__str__()

    def get_leaves(self):
        """Return all leaves in the tree."""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Update bounds for all nodes in the tree."""
        self.root.update_bounds_below()

    def update_predict(self):
        """Create a vectorized prediction function for the tree."""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.sum(
            [leaf.value * leaf.indicator(A).astype(int) for leaf in leaves],
            axis=0
        )

    def pred(self, x):
        """Predict the output for one sample using recursion."""
        return self.root.pred(x)

    def fit(self, explanatory, target, verbose=0):
        """Fit the decision tree to training data."""
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion
        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype='bool')

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}
    - Accuracy on training data : {self.accuracy(self.explanatory,
                                                 self.target)}""")

    def np_extrema(self, arr):
        """Return min and max of an array."""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Randomly select a feature and threshold to split a node."""
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population]
            )
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1-x)*feature_min + x*feature_max
        return feature, threshold

    def fit_node(self, node):
        """Recursively split the dataset to build the tree."""
        node.feature, node.threshold = self.split_criterion(node)

        left_population = (
            node.sub_population &
            (self.explanatory[:, node.feature] > node.threshold)
        )
        right_population = (
            node.sub_population &
            (self.explanatory[:, node.feature] <= node.threshold)
        )

        n_left = left_population.sum()
        pure_left = (np.unique(self.target[left_population]).size <= 1)
        depth_limit = (node.depth + 1 == self.max_depth)
        is_left_leaf = (n_left < self.min_pop) or depth_limit or pure_left

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        n_right = right_population.sum()
        pure_right = (np.unique(self.target[right_population]).size <= 1)
        is_right_leaf = (n_right < self.min_pop) or depth_limit or pure_right

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """Create a leaf node with the majority class of the subset."""
        y_sub = self.target[sub_population]
        if y_sub.size == 0:
            y_parent = self.target[node.sub_population]
            if y_parent.size == 0:
                vals, counts = np.unique(self.target, return_counts=True)
            else:
                vals, counts = np.unique(y_parent, return_counts=True)
        else:
            vals, counts = np.unique(y_sub, return_counts=True)

        value = vals[counts.argmax()]
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.subpopulation = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Create a new internal node as a child."""
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        """Return accuracy of the tree on test data."""
        return np.sum(
            np.equal(self.predict(test_explanatory), test_target)
        ) / test_target.size
