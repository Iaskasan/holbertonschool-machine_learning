#!/usr/bin/env python3
"""
Decision tree implementation using Node and Leaf classes.

This module defines the core data structures used to build and
analyze a decision tree, including recursive depth computation.
"""
import numpy as np


class Node:
    """
    Represents an internal node in a decision tree.

    A Node contains a feature and threshold used for splitting,
    as well as references to its left and right child nodes.
    It also stores its depth within the tree.
    """
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """
        Initialize a decision tree node.

        Args:
            feature: The feature index used for splitting at this node.
            threshold: The threshold value used for splitting.
            left_child (Node): Left child node.
            right_child (Node): Right child node.
            is_root (bool): True if this node is the root of the tree.
            depth (int): Depth of the node in the tree.
        """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth
        self.upper = {}
        self.lower = {}

    def max_depth_below(self):
        """
        Compute the maximum depth reachable below this node.

        This method recursively traverses the subtree rooted at
        this node and returns the greatest depth value found.

        Returns:
            int: Maximum depth in the subtree rooted at this node.
        """
        if self.left_child is None and self.right_child is None:
            return self.depth
        if self.left_child:
            left_depth = self.left_child.max_depth_below()
        if self.right_child:
            right_depth = self.right_child.max_depth_below()
        return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        """
        Count the number of nodes in the subtree rooted at this node.

        If only_leaves is True, only leaf nodes are counted.
        Otherwise, all nodes in the subtree are counted.

        Args:
            only_leaves (bool): Whether to count only leaf nodes.

        Returns:
            int: Number of nodes below this node.
        """
        if self.left_child is None and self.right_child is None:
            return 1
        total = 0
        if self.left_child is not None:
            total += self.left_child.count_nodes_below(only_leaves=only_leaves)
        if self.right_child is not None:
            total += self.right_child.count_nodes_below(
                only_leaves=only_leaves)
        if not only_leaves:
            total += 1
        return total

    def __str__(self):
        """
        Return a formatted string representation of the subtree.

        This method produces a multi-line, human-readable representation
        of the tree structure starting from this node, using ASCII
        connectors to visualize parent-child relationships.

        Returns:
            str: Formatted representation of the subtree.
        """
        return "\n".join(self._str_lines()) + "\n"

    def _str_lines(self, prefix=""):
        """
        Recursively build the formatted lines of the subtree.

        Each call returns a list of strings representing the subtree
        rooted at this node. The prefix is used to align children and
        draw vertical connectors between sibling branches.

        Args:
            prefix (str): Prefix used to align child nodes visually.

        Returns:
            list[str]: Lines representing the formatted subtree.
        """
        lines = [self._label()]
        children = []
        if self.left_child is not None:
            children.append(self.left_child)
        if self.right_child is not None:
            children.append(self.right_child)
        for i, child in enumerate(children):
            is_last = (i == len(children) - 1)
            connector = "+---> "
            lines.append(prefix + connector + child._label())
            if isinstance(child, Node) and (child.left_child
                                            is not None or
                                            child.right_child is not None):
                next_prefix = prefix + ("      " if is_last else "|     ")
                lines.extend(child._str_lines(prefix=next_prefix)[1:])
        return lines

    def _label(self):
        """
        Generate a label for visualization purposes.

        Returns:
            str: Label string for the node.
        """
        name = "root" if self.is_root else "node"
        return f"{name} [feature={self.feature}, threshold={self.threshold}]"

    def get_leaves_below(self):
        """
        Retrieve all leaf nodes in the subtree rooted at this node.

        Returns:
            list[Leaf]: List of leaf nodes in the subtree.
        """
        leaves = []
        if self.left_child is not None:
            leaves.extend(self.left_child.get_leaves_below())
        if self.right_child is not None:
            leaves.extend(self.right_child.get_leaves_below())
        return leaves

    def update_bounds_below(self):
        """
        Recursively propagate lower/upper bound constraints to children.

        Convention used by this project:
          - left_child corresponds to X[feature] > threshold
          - right_child corresponds to X[feature] <= threshold
        """
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1*np.inf}

        f = self.feature
        t = self.threshold

        if self.left_child is not None:
            self.left_child.lower = self.lower.copy()
            self.left_child.upper = self.upper.copy()
            prev = self.left_child.lower.get(f, -np.inf)
            self.left_child.lower[f] = max(prev, t)

        if self.right_child is not None:
            self.right_child.lower = self.lower.copy()
            self.right_child.upper = self.upper.copy()
            prev = self.right_child.upper.get(f, np.inf)
            self.right_child.upper[f] = min(prev, t)

        for child in (self.left_child, self.right_child):
            if child is not None:
                child.update_bounds_below()

    def update_indicator(self):
        """
        Define the indicator function for this node.

        The indicator function takes a data matrix x and returns a boolean
        array indicating which individuals satisfy the node constraints.
        """

        def is_large_enough(x):
            if len(self.lower) == 0:
                return np.ones(x.shape[0], dtype=bool)

            return np.all(
                np.array([x[:, f] > lb for f, lb in self.lower.items()]),
                axis=0
            )

        def is_small_enough(x):
            if len(self.upper) == 0:
                return np.ones(x.shape[0], dtype=bool)

            return np.all(
                np.array([x[:, f] <= ub for f, ub in self.upper.items()]),
                axis=0
            )

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]),
            axis=0
        )

    def pred(self, x):
        """
        Predict the output for a single data point x.
        Args:
            x (np.ndarray): 1D array representing a single data point.
        Returns:
            Predicted value for the data point x.
        """
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):
    """
    Represents a leaf node in a decision tree.

    A Leaf stores a prediction value and has no children.
    """
    def __init__(self, value, depth=None):
        """
        Initialize a leaf node.

        Args:
            value: The predicted value stored in the leaf.
            depth (int): Depth of the leaf in the tree.
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Return the depth of the leaf.

        Returns:
            int: Depth of the leaf node.
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        Count the number of nodes in the subtree rooted at this leaf.

        Args:
            only_leaves (bool): Whether to count only leaf nodes.

        Returns:
            int: Number of nodes in the subtree rooted at this leaf.
        """
        return 1

    def __str__(self):
        """
        Provide a string representation of the leaf node.
        """
        return (f"-> leaf [value={self.value}]")

    def _label(self):
        """
        Return the display label for this leaf.

        Returns:
            str: A one-line string describing the leaf.
        """
        return f"leaf [value={self.value}]"

    def get_leaves_below(self):
        """
        Retrieve all leaf nodes in the subtree rooted at this leaf.
        Returns:
            list[Leaf]: List of leaf nodes in the subtree.
        """
        return [self]

    def update_bounds_below(self):
        """Leaf has no children to update"""
        pass

    def pred(self, x):
        """
        Predict the output for a single data point x.
        Args:
            x (np.ndarray): 1D array representing a single data point.
        Returns:
            Predicted value for the data point x.
        """
        return self.value


class Decision_Tree:
    """
    Decision tree model.

    This class manages the root node and global tree parameters,
    and provides utilities such as computing the tree depth.
    """
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """
        Initialize a decision tree.

        Args:
            max_depth (int): Maximum allowed depth of the tree.
            min_pop (int): Minimum population required to split a node.
            seed (int): Random seed for reproducibility.
            split_criterion (str): Criterion used to split nodes.
            root (Node): Optional root node for the tree.
        """
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
        """
        Compute the maximum depth of the decision tree.

        Returns:
            int: Maximum depth of the tree.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Count the number of nodes in the decision tree.
        Args:
            only_leaves (bool): Whether to count only leaf nodes.
        Returns:
            int: Number of nodes in the tree.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """
        Provide a string representation of the decision tree.
        """
        return self.root.__str__()

    def get_leaves(self):
        """
        Retrieve all leaf nodes in the decision tree.

        Returns:
            list[Leaf]: List of all leaf nodes in the tree.
        """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """
        Initialize and propagate feature bounds from the root to all nodes.
        """
        self.root.update_bounds_below()

    def update_predict(self):
        """
        Build the prediction function for the decision tree.

        After calling this method, self.predict(A) returns a 1D NumPy array
        of predictions for the individuals in A.
        """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        def predict(A):
            n = A.shape[0]
            y_pred = np.empty(n, dtype=int)
            for leaf in leaves:
                mask = leaf.indicator(A)
                y_pred[mask] = leaf.value
            return y_pred
        self.predict = predict

    def pred(self, x):
        """
        Predict the output for a single data point x.
        Args:
            x (np.ndarray): 1D array representing a single data point.
        Returns:
            Predicted value for the data point x.
        """
        return self.root.pred(x)

    def fit_node(self, node):
        """ Recursively fit the subtree rooted at 'node' """
        node.feature, node.threshold = self.split_criterion(node)
        pop = node.sub_population
        left_population = pop & (
            self.explanatory[:, node.feature] > node.threshold
        )
        right_population = pop & (
            self.explanatory[:, node.feature] <= node.threshold
        )
        is_left_leaf = (
            np.sum(left_population) < self.min_pop or
            node.depth + 1 >= self.max_depth or
            np.unique(self.target[left_population]).size == 1
        )

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)
        is_right_leaf = (
            np.sum(right_population) < self.min_pop or
            node.depth + 1 >= self.max_depth or
            np.unique(self.target[right_population]).size == 1
        )

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """ Create and return a new Leaf as child of 'node' """
        values, counts = np.unique(
            self.target[sub_population], return_counts=True
        )
        value = values[np.argmax(counts)]

        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def fit(self, explanatory, target, verbose=0):
        """
        Fit the decision tree to the training data.
        """
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
- Accuracy on training data : {self.accuracy(
                                self.explanatory, self.target)}""")

    def np_extrema(self, arr):
        """
        Compute the minimum and maximum of a NumPy array.
        """
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """ Randomly choose a feature and threshold to split on """
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population])
            diff = feature_max-feature_min
        x = self.rng.uniform()
        threshold = (1-x)*feature_min + x*feature_max
        return feature, threshold

    def get_node_child(self, node, sub_population):
        """ Create and return a new Node as child of 'node' """
        n = Node()
        n.depth = node.depth+1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        """ Compute accuracy on test data """
        return np.sum(
            np.equal(self.predict(test_explanatory),
                     test_target))/test_target.size
