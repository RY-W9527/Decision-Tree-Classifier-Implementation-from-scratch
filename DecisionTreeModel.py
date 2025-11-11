import matplotlib.pyplot as plt
import numpy as np
import copy
from DecisionTreeVisualizer import DecisionTreeVisualizer

class DecisionTreeModel:
    def __init__(self):
        self.tree = None
        self.id2node = {}
        self.depth = 0
        self.visualizer = DecisionTreeVisualizer()

    def predict(self, data):
        """Predict the labels for the given data using the trained decision tree.

        Args:
            data (numpy.ndarray): shape (n_samples, n_features+1(the label))
        Returns:
            numpy.ndarray: Predicted labels, shape (n_samples,)
        """
        assert self.tree is not None, "Decision tree not fitted. Call fit(...) first."
        predictions = []
        for sample in data:
            x_nfeatures = sample[:-1] # exclude the label
            node = self.tree # start from the root
            while not node["leaf"]: # traverse until a leaf node is reached
                if x_nfeatures[node["attr"]] < node["value"]: # perform the split test
                    node = node["left"]
                else:
                    node = node["right"]
            predictions.append(node["label"]) # append the predicted label
        return np.array(predictions, dtype=int)
        
    def prune(self, train_data, val_data: np.ndarray):
        """
        Reduced-error pruning:
        - Only prune internal nodes whose two children are leaves.
        - Candidate leaf label = majority of *training* samples that reach that node.
        - Prune if validation error (on val samples reaching the node) does not increase.
        - Repeat passes until no change.
        Data shape: (n_samples, n_features + 1), last column is label.
        """
        assert self.tree is not None, "Decision tree not fitted. Call fit(...) first."
        self.id2node = {} # reset

        def route(node, data):
            l_dataset = data[data[:, node['attr']] < node['value']]
            r_dataset = data[data[:, node['attr']] >= node['value']]
            return l_dataset, r_dataset

        def traverse_and_prune(node, train_data, val_data):
            # Base case: if the node is a leaf or no training data
            self.id2node[id(node)] = node
            if node['leaf']:
                return node, 1
            
            # Recursive case: traverse left and right subtrees
            left_t, right_t = route(node, train_data)
            left_v, right_v = route(node, val_data)
            node['left'], l_depth = traverse_and_prune(node["left"], left_t, left_v)
            node['right'], r_depth = traverse_and_prune(node["right"], right_t, right_v)
            depth = max(l_depth, r_depth) + 1
            
            # Check if both children are leaves
            if node["left"]["leaf"] and node["right"]["leaf"]:
                # A leaf predicts all samples as its label if it reaches to the leaf
                correct_predictions_before = 0
                if len(left_v): correct_predictions_before += np.sum(left_v[:,-1]==node['left']['label'])
                if len(right_v): correct_predictions_before += np.sum(right_v[:,-1]==node['right']['label'])
                
                # Use the major label of train subset to infer node label
                node_label = self.find_majority_label(train_data)
                # Check if there is validation subset at node; use pruned correction if no data available 
                correct_predictions_after = np.sum(val_data[:,-1]==node_label) 

                # Decide whether to keep the pruning: Prune only if accuracy does not decrease
                if correct_predictions_after >= correct_predictions_before:
                    self.id2node.pop(id(node['right']), None)
                    self.id2node.pop(id(node['left']), None)
                    node.update({"leaf": True, 
                                "label": node_label, 
                                "left": None, 
                                "right": None
                                })
                    depth = 1 
            return node, depth
            
        self.tree, self.depth = traverse_and_prune(self.tree, train_data, val_data)
        return self.tree, self.depth
        
        
                 
    def find_majority_label(self, data):
        if data.ndim == 1:
            labels, counts = np.unique(data.astype(int), return_counts=True)       
        else:
            labels, counts = np.unique(data[:, -1].astype(int), return_counts=True)
        
        majority_label = labels[np.argmax(counts)]
        return majority_label



    # Algorithm follow pseudocode
    def decision_tree_learning(self, data, depth=0):
        """Decision Tree Learning: Fit the decision tree model to the data.

        Args:
            data (the training set): shape (n_samples, n_features + 1(the label))
            depth (int, optional): The depth of the current opperation, also the recursive index. Defaults to 0.

        Returns:
            tuple: (the decision tree, depth of the tree)
        """
        if depth == 0: self.id2node = {}  # Reset registry at root

        if self.is_leaf(data):
            leaf = {"leaf": True, "label": np.unique(data[:, -1])[0]}
            self.id2node[id(leaf)] = leaf # for plotting
            return leaf, depth
        else:
            best_attr, best_value, best_gain = self.find_split(data)
            # Return a leaf node if no valid split is found
            if best_attr is None or best_value is None or best_gain <= 0:
                leaf ={"leaf": True, "label": self.find_majority_label(data)} # majority class
                self.id2node[id(leaf)] = leaf
                return leaf, depth
            node = {"attr": best_attr, "value": best_value, 
                    "left": None, "right": None, "leaf": False}
            self.id2node[id(node)] = node
            # Split the dataset according to the best attribute and value
            l_dataset = data[data[:, best_attr] < best_value]
            r_dataset = data[data[:, best_attr] >= best_value]
            # Recursively build the left and right subtrees
            node["left"], l_depth = self.decision_tree_learning(l_dataset, depth+1)
            node["right"], r_depth = self.decision_tree_learning(r_dataset, depth+1)
            # Update the tree and depth class attributes
            self.tree = node
            self.depth = max(l_depth, r_depth)
            return self.tree, self.depth


    def find_split(self, data):
        ''' An efficient method for finding good split points is to sort the values of the attribute, 
        and then consider only split points that are between two examples in sorted order, 
        while keeping track of the running totals of examples of each class for each side of the split point.
        '''
        best_gain = -1
        best_attr = None
        best_value = None

        n_features = data.shape[1] - 1  

        for attribute in range(n_features):
            sorted_data = data[data[:, attribute].argsort()]
            X = sorted_data[:, attribute]
            y = sorted_data[:, -1]

            for i in range(len(X) - 1):
                if y[i] != y[i + 1]: 
                    split_value = (X[i] + X[i + 1]) / 2

                    S_left = sorted_data[sorted_data[:, attribute] < split_value]
                    S_right = sorted_data[sorted_data[:, attribute] >= split_value]

                    if len(S_left) == 0 or len(S_right) == 0:
                        continue

                    gain = self.calculate_gain(sorted_data, S_left, S_right)

                    if gain > best_gain:
                        best_gain = gain
                        best_attr = attribute
                        best_value = split_value

        return best_attr, best_value, best_gain


    def is_leaf(self, data):
        '''Checks if a node should STOP splitting'''
        if len(np.unique(data[:, -1])) == 1:
            return True
        if data.shape[0] == 0:
            return True
        return False


    def calculate_entropy(self, data):
        '''Calculate the information entropy'''
        y = data[:, -1]
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities))


    def calculate_remainder(self, S_left, S_right):
        '''Calculate the remainder'''
        total = S_left.shape[0] + S_right.shape[0]
        if total == 0:
            return 0
        return (S_left.shape[0]/total)*self.calculate_entropy(S_left) + \
            (S_right.shape[0]/total)*self.calculate_entropy(S_right)


    def calculate_gain(self, data, S_left, S_right):
        '''Calculate the information gain'''
        if len(S_left) == 0 or len(S_right) == 0:
            return 0
        return self.calculate_entropy(data) - self.calculate_remainder(S_left, S_right)

    def plot(self, save_path=None, ax=None, fontsize=8, node_pad=0.3):
        # Plot with Decision Tree Visualizer which does not make nodes overlap
        self.visualizer.update(tree=self.tree, id2node=self.id2node, depth=self.depth)
        self.visualizer.plot(save_path, 
                             num_lvl=6, # plot to the 7th depth 
                             ax=ax,
                             fontsize=fontsize,
                             node_pad=node_pad,
                             )
       