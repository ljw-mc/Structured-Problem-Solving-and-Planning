import numpy as np
# DO NOT ADD TO OR MODIFY ANY IMPORT STATEMENTS

def find_number_of_examples(examples):
    return len(examples)

def find_partition_number(examples, goal, col_idx):
    classes = goal[1]
    partitions = {}

    for example in examples:
        if example[col_idx] not in partitions.keys():
            partitions[example[col_idx]] = 1
        else:
            partitions[example[col_idx]] += 1
    return partitions

def split_examples(exmaples, attribute, col_idx):
    classes = attribute[1]
    split = {}

    for example in examples:
        if example[col_idx] not in split.keys():
            split[example[col_idx]] = [example]
        else:
            split[example[col_idx]].append(example)
    
    return split


def dataset_partition(examples, goal):
    classes = goal[1]
    partitions = {}

    for example in examples:
        if example[-1] not in partitions.keys():
            partitions[example[-1]] = [example]
        else:
            partitions[example[-1]].append(example)
    
    return partitions

def cond_prob(dataset_split, attribute, col_idx, p_y, partition, goal):
    p_xy = {}
    
    fxy = {}
    for y in range(len(p_y)):
        for x in range(len(goal[1])):
            fxy[str([x, y])] = 0
    
    for x in dataset_split.keys():
        for example in dataset_split[x]:
            fxy[str([x, example[col_idx]])] += 1
    
    for y in range(len(p_y)):
        for x in range(len(goal[1])):
            fxy[str([x, y])] = fxy[str([x, y])] / partition[y]


    return fxy

def dt_entropy(goal, examples):
    """
    Compute entropy over discrete random varialbe for decision trees.
    Utility function to compute the entropy (wich is always over the 'decision'
    variable, which is the last column in the examples).

    :param goal: Decision variable (e.g., WillWait), cell array.
    :param examples: Training data; the final class is given by the last column.
    :return: Value of the entropy of the decision variable, given examples.
    """
    # INSERT YOUR CODE HERE.
    entropy = 0.
    # Be careful to check the number of examples
    # Avoid NaN examples by treating the log2(0.0) = 0
    number_of_examples = find_number_of_examples(examples)
    partition = find_partition_number(examples, goal, col_idx=-1)

    for c in range(len(goal[1])):
        if partition[c] == 0:
            continue
        else:
            p = partition[c] / number_of_examples
            entropy += p * np.log2(1/p)

    return entropy


def dt_cond_entropy(attribute, col_idx, goal, examples):
    """
    Compute the conditional entropy for attribute. Utility function to compute the conditional entropy (which is always
    over the 'decision' variable or goal), given a specified attribute.

    :param attribute: Dataset attribute, cell array.
    :param col_idx: Column index in examples corresponding to attribute.
    :param goal: Decision variable, cell array.
    :param examples: Training data; the final class is given by the last column.
    :return: Value of the conditional entropy, given the attribute and examples.
    """
    # INSERT YOUR CODE HERE.
    cond_entropy = 0.0
    tot_num_examples = find_number_of_examples(examples)
    split = find_partition_number(examples, attribute, col_idx)
    dataset_split = dataset_partition(examples, attribute)
    
    p_y = {} # p(y) prior info - ie attribute info
    for c in range(len(attribute[1])):
        p_y[c] = split[c] / tot_num_examples

    px_y = cond_prob(dataset_split, attribute, col_idx, p_y, split, goal) # p(x|y)

    H = [0] * len(attribute[1])
    for y in range(len(p_y)):
        for x in range(len(goal[1])):
            # H(x|y) = sum p(x|y) log2 (1/ p(x|y))
            if px_y[str([x,y])] == 0:
                continue
            else:
                H[y] += px_y[str([x,y])] * np.log2(1/px_y[str([x,y])])
    
    for y in range(len(p_y)):
        cond_entropy += (p_y[y] * H[y])





    return cond_entropy


def dt_info_gain(attribute, col_idx, goal, examples):
    """
    Compute information gain for attribute.
    Utility function to compute the information gain after splitting on attribute.

    :param attribute: Dataset attribute, cell array.
    :param col_idx: Column index in examples corresponding to attribute.
    :param goal: Decision variable, cell array.
    :param examples: Training data; the final class is given by the last column.
    :return: Value of the information gain, given the attribute and examples.

    """
    # INSERT YOUR CODE HERE.
    info_gain = 0.
    h_of_x = dt_entropy(goal, examples)
    h_of_x_given_y = dt_cond_entropy(attribute, col_idx, goal, examples)
    info_gain = h_of_x - h_of_x_given_y

    return info_gain


def dt_intrinsic_info(attribute, col_idx, examples):
    """
    Compute the intrinsic information for attribute.
    Utility function to compute the intrinsic information of a specified attribute.

    :param attribute: Dataset attribute, cell array.
    :param col_idx: Column index in examples corresponding to attribute.
    :param examples: Training data; the final class is given by the last column.
    :return: Value of the intrinsic information for the attribute and examples.
    """
    # INSERT YOUR CODE HERE.
    # Be careful to check the number of examples
    # Avoid NaN examples by treating the log2(0.0) = 0
    # Get the partition of the attribute
    partition = find_partition_number(examples, attribute, col_idx)
    
    # Compute the total number of examples
    total_examples = find_number_of_examples(examples)
    
    # Compute the intrinsic information
    intrinsic_info = 0.0
    for value in partition.values():
        if value > 0:
            p = value / total_examples
            intrinsic_info -= p * np.log2(p)
    
    return intrinsic_info



def dt_gain_ratio(attribute, col_idx, goal, examples):
    """
    Compute information gain ratio for attribute.
    Utility function to compute the gain ratio after splitting on attribute. Note that this is just the information
    gain divided by the intrinsic information.
    :param attribute: Dataset attribute, cell array.
    :param col_idx: Column index in examples corresponding to attribute.
    :param goal: Decision variable, cell array.
    :param examples: Training data; the final class is given by the last column.
    :return: Value of the gain ratio, given the attribute and examples.
    """
    # INSERT YOUR CODE HERE.
    # Avoid NaN examples by treating 0.0/0.0 = 0.0
    gain_ratio = 0
    
    # Compute information gain and intrinsic information
    info_gain = dt_info_gain(attribute, col_idx, goal, examples)
    intrinsic_info = dt_intrinsic_info(attribute, col_idx, examples)
    
    # Avoid division by zero error
    if intrinsic_info == 0:
        return 0
    
    # Compute gain ratio
    gain_ratio = info_gain / intrinsic_info
    
    return gain_ratio

def is_empty(list):
    return (len(list)) == 0

def same_class(examples, goal):
    # Get the class label of the first example
    first_label = examples[0][-1]

    # Check whether all other examples have the same class label
    for example in examples[1:]:
        if example[-1] != first_label:
            return False

    return True

def get_smallest_key_max_value(d):
    max_value = max(d.values())
    keys_with_max_value = [k for k, v in d.items() if v == max_value]
    smallest_key = min(keys_with_max_value)
    return smallest_key

def find_best_attribute(attributes, examples, goal, score_fun):
    importance = {}
    bob = {}
    for col_idx in range(len(attributes)):
        todd = attributes[col_idx]
        importance[col_idx] = score_fun(attributes[col_idx], col_idx, goal, examples)

    return get_smallest_key_max_value(importance)

class TreeNode:
    """
    Class representing a node in a decision tree.
    When parent == None, this is the root of a decision tree.
    """
    def __init__(self, parent, attribute, examples, is_leaf, label):
        # Parent node in the tree
        self.parent = parent
        # Attribute that this node splits on
        self.attribute = attribute
        # Examples used in training
        self.examples = examples
        # Boolean representing whether this is a leaf in the decision tree
        self.is_leaf = is_leaf
        # Label of this node (important for leaf nodes that determine classification output)
        self.label = label
        # List of nodes
        self.branches = []

    def query(self, attributes: np.ndarray, goal, query: np.ndarray) -> (int, str):
        """
        Query the decision tree that self is the root of at test time.

        :param attributes: Attributes available for splitting at this node
        :param goal: Goal, decision variable (classes/labels).
        :param query: A test query which is a (n,) array of attribute values, same format as examples but with the final
                      class label).
        :return: label_val, label_txt: integer and string representing the label index and label name.
        """
        node = self
        while not node.is_leaf:
            b = node.get_branch(attributes, query)
            node = node.branches[b]

        return node.label, goal[1][node.label]

    def get_branch(self, attributes: list, query: np.ndarray):
        """
        Find attributes in a set of attributes and determine which branch to use (return index of that branch)

        :param attributes: list of attributes
        :param query: A test query which is a (n,) array of attribute values.
        :return:
        """
        for i in range(len(attributes)):
            todd = self.attribute[0]
            jobb = attributes[i][0]
            if self.attribute[0] == attributes[i][0]:
                return query[i]
        # Return None if that attribute can't be found
        return None

    def count_tree_nodes(self, root=True) -> int:
        """
        Count the number of decision nodes in a decision tree.
        :param root: boolean indicating if this is the root of a decision tree (needed for recursion base case)
        :return: number of nodes in the tree
        """
        num = 0
        for branch in self.branches:
            num += branch.count_tree_nodes(root=False) + 1
        return num + root

def get_examples_with_attribute_value(examples, best_index, value):
    return [example for example in examples if example[best_index] == value]

def learn_decision_tree(parent, attributes, goal, examples, score_fun):
    """
    Recursively learn a decision tree from training data.
    Learn a decision tree from training data, using the specified scoring function to determine which attribute to split
    on at each step. This is an implementation of the algorithm on pg. 702 of AIMA.

    :param parent: Parent node in tree (or None if first call of this algorithm).
    :param attributes: Attributes avaialble for splitting at this node.
    :param goal: Goal, decision variable (classes/labels).
    :param examples: Subset of examples that reach this point in the tree.
    :param score_fun: Scoring function used (dt_info_gain or dt_gain_ratio)
    :return: Root node of tree structure.
    """

    name_of_attributes = []
    for attribute in attributes:
        name_of_attributes.append(attribute[0])
    # YOUR CODE GOES HERE
    node = TreeNode(parent, name_of_attributes, examples=examples, is_leaf=False, label=1)
    # 1. Do any examples reach this point?
    if is_empty(examples):
        return TreeNode(parent, name_of_attributes, examples, is_leaf=True, label=plurality_value(goal, parent))

    # 2. Or do all examples have the same class/label? If so, we're done!
    if same_class(examples, goal):
        return TreeNode(parent, name_of_attributes, examples, is_leaf=True, label=examples[0][-1])

    # 3. No attributes left? Choose the majority class/label.
    if is_empty(attributes):
        return TreeNode(parent, attribute=name_of_attributes, examples=examples, is_leaf=True, label=plurality_value(goal, examples))
    
    # 4. Otherwise, need to choose an attribute to split on, but which one? Use score_fun and loop over attributes!
    else:
        # Best score?
        best_index = find_best_attribute(attributes=attributes, examples=examples, goal=goal, score_fun=score_fun)
        new_attributes = attributes.copy()
        #new_attributes.remove(attributes[best_index])
        number_of_features = len(attributes[best_index])
        for value in range(number_of_features+1):
            exs = get_examples_with_attribute_value(examples, best_index, value)
            new_node = learn_decision_tree(examples, attributes=new_attributes, goal=goal, examples=exs, score_fun=score_fun)
            node.branches.append(new_node)


        # NOTE: to pass the Autolab tests, when breaking ties you should always select the attribute with the smallest (i.e.
        # leftmost) column index!

        # Create a new internal node using the best attribute, something like:
        # node = TreeNode(parent, attributes[best_index], examples, False, 0)

        # Now, recurse down each branch (operating on a subset of examples below).
        # You should append to node.branches in this recursion

    return node



def plurality_value(goal: tuple, examples: np.ndarray) -> int:
    """
    Utility function to pick class/label from mode of examples (see AIMA pg. 702).
    :param goal: Tuple representing the goal
    :param examples: (n, m) array of examples, each row is an example.
    :return: index of label representing the mode of example labels.
    """
    vals = np.zeros(len(goal[1]))

    # Get counts of number of examples in each possible attribute class first.
    for i in range(len(goal[1])):
        vals[i] = sum(examples[:, -1] == i)

    return np.argmax(vals)


if __name__ == '__main__':
    # Example use of a decision tree from AIMA's restaurant problem on page (pg. 698)
    # Each attribute is a tuple of 2 elements: the 1st is the attribute name (a string), the 2nd is a tuple of options
    a0 = ('Alternate', ('No', 'Yes'))
    a1 = ('Bar', ('No', 'Yes'))
    a2 = ('Fri-Sat', ('No', 'Yes'))
    a3 = ('Hungry', ('No', 'Yes'))
    a4 = ('Patrons', ('None', 'Some', 'Full'))
    a5 = ('Price', ('$', '$$', '$$$'))
    a6 = ('Raining', ('No', 'Yes'))
    a7 = ('Reservation', ('No', 'Yes'))
    a8 = ('Type', ('French', 'Italian', 'Thai', 'Burger'))
    a9 = ('WaitEstimate', ('0-10', '10-30', '30-60', '>60'))
    attributes = [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9]
    # The goal is a tuple of 2 elements: the 1st is the decision's name, the 2nd is a tuple of options
    goal = ('WillWait', ('No', 'Yes'))

    # Let's input the training data (12 examples in Figure 18.3, AIMA pg. 700)
    # Each row is an example we will use for training: 10 features/attributes and 1 outcome (the last element)
    # The first 10 columns are the attributes with 0-indexed indices representing the value of the attribute
    # For example, the leftmost column represents the attribute 'Alternate': 0 is 'No', 1 is 'Yes'
    # Another example: the 3rd last column is 'Type': 0 is 'French', 1 is 'Italian', 2 is 'Thai', 3 is 'Burger'
    # The 11th and final column is the label corresponding to the index of the goal 'WillWait': 0 is 'No', 1 is 'Yes'
    examples = np.array([[1, 0, 0, 1, 1, 2, 0, 1, 0, 0, 1],
                         [1, 0, 0, 1, 2, 0, 0, 0, 2, 2, 0],
                         [0, 1, 0, 0, 1, 0, 0, 0, 3, 0, 1],
                         [1, 0, 1, 1, 2, 0, 1, 0, 2, 1, 1],
                         [1, 0, 1, 0, 2, 2, 0, 1, 0, 3, 0],
                         [0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
                         [0, 1, 0, 0, 0, 0, 1, 0, 3, 0, 0],
                         [0, 0, 0, 1, 1, 1, 1, 1, 2, 0, 1],
                         [0, 1, 1, 0, 2, 0, 1, 0, 3, 3, 0],
                         [1, 1, 1, 1, 2, 2, 0, 1, 1, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
                         [1, 1, 1, 1, 2, 0, 0, 0, 3, 2, 1]])

    # entropy = dt_entropy(goal, examples)
    # ba = dt_cond_entropy(a5, 5, goal, examples)
    # bob = dt_info_gain(a5, 5, goal, examples)

    # lol1 = dt_intrinsic_info(a4, 4, examples)
    # lol2 = new_dt_intrinsic_info(a4, 4, examples)

    # print(lol1, lol2)

    # print("hel")


    # Build your decision tree using dt_info_gain as the score function
    tree = learn_decision_tree(None, attributes, goal, examples, dt_info_gain)
    # Query the tree with an unseen test example: it should be classified as 'Yes'
    test_query = np.array([0, 0, 1, 1, 2, 0, 0, 0, 2, 3])
    _, test_class = tree.query(attributes, goal, test_query)
    print("Result of query: {:}".format(test_class))

    # Repeat with dt_gain_ratio:
    tree_gain_ratio = learn_decision_tree(None, attributes, goal, examples, dt_gain_ratio)
    # Query this new tree: it should also be classified as 'Yes'
    _, test_class = tree_gain_ratio.query(attributes, goal, test_query)
    print("Result of query with gain ratio as score: {:}".format(test_class))
