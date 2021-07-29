"""
A mechanism to predict which variables are used at which point in a decoded
tree.

"""

# Copyright (C) 2021
# Benjamin Paaßen
# The University of Sydney

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2020, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '0.2.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'benjamin.paassen@sydney.edu.au'

from tqdm import tqdm
import copy
import torch
import random
import tree
import numpy as np
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin
import python_ast_utils

def predict_(classifier, x):
    """ An internal auxiliary function to support classifier
    predictions for single data points.

    Parameters
    ----------
    classifier: class SVC or int
        The classifier to be used (either a support vector machine
        or a constant).
    x: ndarray
        A vector to be used as input for the classification.

    Returns
    -------
    j: int
        A single prediction.

    """
    # if the classifier is a constant, use that constant
    if isinstance(classifier, int):
        return classifier
    elif isinstance(classifier, SVC):
        return classifier.predict(np.expand_dims(x, 0)).squeeze()
    else:
        raise ValueError('Unexpected classifier of type %s' % str(classifier))

def decision_function_(classifier, x, ref_length = 0):
    """ An internal auxiliary function to support classifier
    predictions for single data points where we need the value
    of the decision function.

    Parameters
    ----------
    classifier: class SVC or int
        The classifier to be used (either a support vector machine
        or a constant).
    x: ndarray
        A vector to be used as input for the classification.

    Returns
    -------
    y: ndarray
        A vector of decision function values. The regular prediction
        of the classifier would be np.argmax(y).

    """
    # if the classifier is a constant, use that constant
    if isinstance(classifier, int):
        y = np.full(max(ref_length, classifier+1), -np.inf)
        y[classifier] = 1.
    elif isinstance(classifier, SVC):
        yraw = classifier.decision_function(np.expand_dims(x, 0))
        y = np.full(max(ref_length, np.max(classifier.classes_)+1), -np.inf)
        if len(classifier.classes_) == 2:
            y[classifier.classes_[0]] = 0.
            y[classifier.classes_[1]] = yraw[0]
        else:
            y[classifier.classes_] = yraw
    else:
        raise ValueError('Unexpected classifier of type %s' % str(classifier))
    return y

def store_entry_(name, lst, dct):
    """ Check if the given name is already contained in the given
    dictionary. If not, it is appended both to the list and to
    the dictionary and the dictionary maps to the list index.

    Parameters
    ----------
    name: str
        Some string.
    lst: list
        A list of existing strings.
    dct: dict
        A dictionary mapping strings to list indices, i.e.
        list[dct[name]] == name for all entries in dct and
        lst. 

    """ 
    if name in dct:
        return dct[name]
    else:
        i = len(lst)
        dct[name] = i
        lst.append(name)
        return i

class VariableClassifier(BaseEstimator):
    """ Implements a classifier to predict variable references and
    values in an abstract syntax tree. More precisely, this model
    performs the following predictions:

    1. For any FunctionDef and ClassDef it selects a name from the
       training data set.
    2. For any Assignment it decides whether this assignment has an
       already existing or a new variable on the left hand side and,
       if it is a new variable, it selects a name from the training
       data set.
    3. For any Call it decides which function is called from the list
       in python_ast_utils._builtin_funs plus the list of locally
       declared functions.
    4. For any Constant it selects a value from the training data set.

    All decisions are made with a support vector machine with radial
    basis function kernel.

    As input features for the prediction, we use an autoencoder. In
    particular, we encode the tree as a vector, decode it again via
    teacher forcing and use the vectors during decoding as input vectors
    for the variable classifiers.

    Parameters
    ----------
    autoencoder: class rtgae2.RTGAE
        A pre-trained autoencoder for Python programs.
    C: float (default = 100)
        The inverse regularization strength for the SVM used internally
    max_points: int (default = None)
        If given, the SVM classifiers are trained at most on max_points. If
        more training data is available, it is subsampled.

    """
    def __init__(self, autoencoder, C = 100., max_points = None):
        self.autoencoder = autoencoder
        self.C = C
        self.max_points = max_points


    def predict(self, tree, verbose = False):
        """ Predicts the variable references in the given tree and attaches
        them to the tree in place.

        Parameters
        ----------
        tree: class tree.Tree
            The syntax tree of a Python program (without variable references).
        verbose: bool (default = False)
            If set to true, additional info will be printed.

        """
        # encode the current tree to a vector
        h = self.autoencoder.encode(tree)
        # determine the correct rules to decode it
        nont, seq = self.autoencoder.parser_.parse_tree(tree)
        # force-decode the current tree to obtain vectorial encodings
        score_list = []
        decoding_list = []
        self.autoencoder.decode_forced_(h, nont, seq, 0, score_list, decoding_list)
        # then, go in depth first search through the tree and make
        # predictions wherever needed.
        available_functions = copy.copy(python_ast_utils._builtin_funs)
        available_variables = []

        stk = [(tree, False)]
        i = 0
        while stk:
            # get the current syntax subtree and the locally available
            # functions and variables from the stack
            node, in_lhs_assign = stk.pop()
            if verbose:
                if in_lhs_assign:
                    print('processing node %d: %s (in lhs_assign mode)' % (i, node._label))
                else:
                    print('processing node %d: %s' % (i, node._label))

            # get the current vector encoding
            x = decoding_list[i].detach().numpy()
            if node._label == 'FunctionDef':
                # for a function definition, we need to decide on a
                # function name
                if len(self.fun_names_) == 0:
                    name = 'f'
                else:
                    j = predict_(self.cls_fun_name_, x)
                    name = self.fun_names_[j]
                if verbose:
                    print('decided on function name %s' % name)
                node.name = name
                # and we add the function to the list of locally
                # available functions.
                available_functions.append(name)
            elif node._label == 'ClassDef':
                # for a class definitions we need to decide on a
                # class name
                if len(self.class_names_) == 0:
                    name = 'C'
                else:
                    j = predict_(self.cls_class_name_, x)
                    name = self.class_names_[j]
                if verbose:
                    print('decided on class name %s' % name)
                node.name = name
                # and we add the function to the list of locally
                # available functions (because a class instantiation
                # is handled the same as a function call in python).
                available_functions.append(name)
            elif node._label == 'Assign':
                # for an assignment we need to signal to the children
                # that we are now in the left hand side of an assignment
                stk.append((node._children[-1], False))
                for child in node._children[-2::-1]:
                    stk.append((child, True))
                i += 1
                continue
            elif node._label == 'AugAssign' or node._label == 'AnnAssign':
                for child in node._children[:0:-1]:
                    stk.append((child, False))
                stk.append((node._children[0], True))
                i += 1
                continue
            elif node._label == 'For':
                # in a for loop, we definitely declare
                # at least one iteration variable
                if node._children[0]._label == 'Name':
                    if len(node._children[0]._children) > 0:
                        raise ValueError('Iteration variable is not permitted to have children in the syntax tree.')
                    if len(self.var_names_) == 0:
                        name = 'i'
                    else:
                        y = decision_function_(self.cls_var_name_, x, 1 + len(self.var_names_))
                        j = np.argmax(y[1:])
                        name = self.var_names_[j]
                    if verbose:
                        print('decided on iteration variable name %s' % name)
                    node._children[0].id = name
                    # put the new variable onto the list of available variables
                    available_variables.append(name)
                    # advance counter for the Name node
                    i += 1
                elif node._children[0]._label == 'Tuple':
                    # advance counter for the 'Tuple' node
                    i += 1
                    child_idx = 0
                    for it_node in node._children[0]._children:
                        if len(it_node._children) > 0:
                            raise ValueError('Iteration variable is not permitted to have children in the syntax tree.')
                        # advance counter for each child
                        i += 1
                        it_x = decoding_list[i].detach().numpy()
                        if len(self.var_names_) == 0:
                            name = chr(ord('i') + child_idx)
                            child_idx += 1
                        else:
                            y = decision_function_(self.cls_var_name_, it_x, 1 + len(self.var_names_))
                            j = np.argmax(y[1:])
                            name = self.var_names_[j]
                        if verbose:
                            print('decided on iteration variable name %s' % name)
                        it_node.id = name
                        # put the new variable onto the list of available variables
                        available_variables.append(name)
                else:
                    raise ValueError('This classifier can only handle for loops which declare an atomic iteration variable with a single Name node OR a tuple of iteration variables')

                # put the other children onto the stack
                for child in node._children[:0:-1]:
                    stk.append((child, False))
                # advance counter for the 'For' loop
                i += 1
                continue
            elif node._label == 'Call':
                # in a function call we need to identify the function
                # that is called

                # with this classifier, we only handle the case that a function
                # is a direct child of the call. If the function is an attribute
                # of a class, for example, we do not consider this here but
                # instead treat it as a variable
                fun_name_child = node._children[0]
                if fun_name_child._label == 'Name' and len(fun_name_child._children) == 0:
                    # identify the function that is called
                    y = decision_function_(self.cls_fun_, x)
                    j = np.argmax(y[:len(available_functions)])
                    name = available_functions[j]
                    fun_name_child.id = name
                    if verbose:
                        print('decided on function reference %s' % name)
                    # put the other children onto the stack
                    for child in node._children[:0:-1]:
                        stk.append((child, False))
                    i += 2
                    continue
            elif node._label == 'Name':
                if in_lhs_assign:
                    # if we are in the left-hand-side of an assignment,
                    # we need to decide whether we wish to reference an
                    # existing variable or declare a new one. We do so
                    # by asking the variable name classifier
                    j = predict_(self.cls_var_name_, x)
                    if j == 0:
                        # if the predicted variable name is zero, this
                        # means that we shall reference an existing
                        # variable
                        if len(available_variables) == 0:
                            name = 'x'
                        else:
                            y = decision_function_(self.cls_var_, x)
                            j = np.argmax(y[:len(available_variables)])
                            name = available_variables[j]
                    else:
                        # add the name to the list of available
                        # variables
                        name = self.var_names_[j-1]
                        available_variables.append(name)
                else:
                    # select which existing variable shall be referenced
                    if len(available_variables) == 0:
                        name = 'x'
                    else:
                        y = decision_function_(self.cls_var_, x)
                        j = np.argmax(y[:len(available_variables)])
                        name = available_variables[j]
                # store the name in the node information
                node.id = name
                if verbose:
                    print('decided on variable reference %s' % name)
            elif node._label == 'Constant':
                # if we process a Constant, we need to decide the
                # value
                j = predict_(self.cls_val_, x)
                val = self.values_[j]
                node.value = val
                if verbose:
                    print('decided on value %s' % str(val))
            elif node._label == 'arg':
                # if we process an argument, this introduces a new
                # variable
                if len(self.var_names_) == 0:
                    name = 'i'
                else:
                    y = decision_function_(self.cls_var_name_, x, 1 + len(self.var_names_))
                    j = np.argmax(y[1:])
                    name = self.var_names_[j]
                available_variables.append(name)
                node.arg = name
                if verbose:
                    print('decided on argument name %s' % name)
            # after the current node is processed, go on to
            # process the children
            i += 1
            for child in node._children[::-1]:
                stk.append((child, False))


    def fit(self, trees, verbose = False):
        # initialize the training data lists
        classifiers = ['cls_fun_name_', 'cls_class_name_', 'cls_var_name_', 'cls_fun_', 'cls_var_', 'cls_val_']

        training_data = {}
        for cls in classifiers:
            X = []
            Y = []
            training_data[cls] = (X, Y)
        # initialize the list of available function, class,
        # and variable names, as well as values
        self.fun_names_   = []
        self.class_names_ = []
        self.var_names_   = []
        # we initialize the value list with a few defaults that
        # should always be available
        self.values_      = [True, False, 'string', 1]
        # initialize auxiliary dictionaries mapping names to list indices
        fun_dict   = {}
        class_dict = {}
        var_dict   = {}
        val_dict   = {}
        for i in range(len(self.values_)):
            val_dict[self.values_[i]] = i

        # iterate over all trees
        for tree in trees:
            # encode the current tree to a vector
            h = self.autoencoder.encode(tree)
            # determine the correct rules to decode it
            nont, seq = self.autoencoder.parser_.parse_tree(tree)
            # force-decode the current tree to obtain vectorial encodings
            score_list = []
            decoding_list = []
            self.autoencoder.decode_forced_(h, nont, seq, 0, score_list, decoding_list)
            # initialize auxiliary variables
            available_functions = copy.copy(python_ast_utils._builtin_funs)
            available_fun_dict  = {}
            for i in range(len(available_functions)):
                available_fun_dict[available_functions[i]] = i
            available_variables = []
            available_var_dict  = {}

            # then, go in depth first search through the tree and make
            # predictions wherever needed.
            stk = [(tree, False)]
            i = 0
            while stk:
                # get the current syntax subtree and the locally available
                # functions and variables from the stack
                node, in_lhs_assign = stk.pop()

                # get the current vector encoding
                x = decoding_list[i].detach().numpy()
                if node._label == 'FunctionDef':
                    # store the function name
                    j = store_entry_(node.name, self.fun_names_, fun_dict)
                    # store the training data
                    training_data['cls_fun_name_'][0].append(x)
                    training_data['cls_fun_name_'][1].append(j)
                    # and we add the function to the list of locally
                    # available functions.
                    store_entry_(node.name, available_functions, available_fun_dict)
                elif node._label == 'ClassDef':
                    # store the class name
                    j = store_entry_(node.name, self.class_names_, class_dict)
                    # store the training data
                    training_data['cls_class_name_'][0].append(x)
                    training_data['cls_class_name_'][1].append(j)
                    # and we add the class to the list of locally
                    # available functions.
                    store_entry_(node.name, available_functions, available_fun_dict)
                elif node._label == 'Assign':
                    # for an assignment we need to signal to the children
                    # that we are now in the left hand side of an assignment
                    stk.append((node._children[-1], False))
                    for child in node._children[-2::-1]:
                        stk.append((child, True))
                    i += 1
                    continue
                elif node._label == 'AugAssign' or node._label == 'AnnAssign':
                    for child in node._children[:0:-1]:
                        stk.append((child, False))
                    stk.append((node._children[0], True))
                    i += 1
                    continue
                elif node._label == 'For':
                    # in a for loop, we definitely declare
                    # at least one iteration variable
                    if node._children[0]._label == 'Name':
                        if len(node._children[0]._children) > 0:
                            raise ValueError('Iteration variable is not permitted to have children in the syntax tree.')
                        # store the variable name
                        name = node._children[0].id
                        j = store_entry_(name, self.var_names_, var_dict)
                        # store the training data
                        training_data['cls_var_name_'][0].append(x)
                        training_data['cls_var_name_'][1].append(j+1)
                        # and we add the variable to the list of locally
                        # available variables.
                        store_entry_(name, available_variables, available_var_dict)
                        i += 1
                    elif node._children[0]._label == 'Tuple':
                        # advance counter for the 'Tuple' node
                        i += 1
                        for it_node in node._children[0]._children:
                            if len(it_node._children) > 0:
                                raise ValueError('Iteration variable is not permitted to have children in the syntax tree.')
                            # advance counter for each child
                            i += 1
                            it_x = decoding_list[i].detach().numpy()
                            # store the new variable names
                            name = it_node.id
                            j = store_entry_(name, self.var_names_, var_dict)
                            # store the training data
                            training_data['cls_var_name_'][0].append(it_x)
                            training_data['cls_var_name_'][1].append(j+1)
                            # and we add the variable to the list of locally
                            # available variables.
                            store_entry_(name, available_variables, available_var_dict)
                    else:
                        raise ValueError('This classifier can only handle for loops which declare an atomic iteration variable with a single Name node OR a tuple of iteration variables')

                    # put the other children onto the stack
                    for child in node._children[:0:-1]:
                        stk.append((child, False))
                    # advance counter for the 'For' loop
                    i += 1
                    continue
                elif node._label == 'Call':
                    # in a function call we need to identify the function
                    # that is called

                    # with this classifier, we only handle the case that a function
                    # is a direct child of the call. If the function is an attribute
                    # of a class, for example, we do not consider this here but
                    # instead treat it as a variable
                    fun_name_child = node._children[0]
                    if fun_name_child._label == 'Name' and len(fun_name_child._children) == 0:
                        name = fun_name_child.id
                        # store the training data
                        j = available_fun_dict.get(name)
                        if j is not None:
                            training_data['cls_fun_'][0].append(x)
                            training_data['cls_fun_'][1].append(j)
                        # put the other children onto the stack
                        for child in node._children[:0:-1]:
                            stk.append((child, False))
                        i += 2
                        continue
                elif node._label == 'Name':
                    name = node.id
                    if in_lhs_assign:
                        # if we are in the left-hand-side of an assignment,
                        # check if we declare a new variable or re-use an
                        # old one
                        j = available_var_dict.get(name)
                        if j is None:
                            # if we declare a new one, store the corresponding
                            # training data
                            j = store_entry_(name, self.var_names_, var_dict)
                            training_data['cls_var_name_'][0].append(x)
                            training_data['cls_var_name_'][1].append(j+1)
                            # and we add the variable to the list of locally
                            # available variables.
                            store_entry_(name, available_variables, available_var_dict)
                        else:
                            # if we re-use an existing one, store the
                            # corresponding training data
                            training_data['cls_var_name_'][0].append(x)
                            training_data['cls_var_name_'][1].append(0)
                            training_data['cls_var_'][0].append(x)
                            training_data['cls_var_'][1].append(j)
                    else:
                        j = available_var_dict.get(name)
                        if j is not None:
                            training_data['cls_var_'][0].append(x)
                            training_data['cls_var_'][1].append(j)
                elif node._label == 'Constant':
                    # store the training data for the value classifier
                    j = store_entry_(node.value, self.values_, val_dict)
                    training_data['cls_val_'][0].append(x)
                    training_data['cls_val_'][1].append(j)
                elif node._label == 'arg':
                    # if we process an argument, this introduces a new
                    # variable
                    name = node.arg
                    j = store_entry_(name, self.var_names_, var_dict)
                    training_data['cls_var_name_'][0].append(x)
                    training_data['cls_var_name_'][1].append(j+1)
                    # and we add the variable to the list of locally
                    # available variables.
                    store_entry_(name, available_variables, available_var_dict)
                # after the current node is processed, go on to
                # process the children
                i += 1
                for child in node._children[::-1]:
                    stk.append((child, False))

        if verbose:
            print('collected %d function names, %d class names, %d variable names, and %d values' % (len(self.fun_names_), len(self.class_names_), len(self.var_names_), len(self.values_)))

        # after we have collected all training data, train all classifiers
        for cls in classifiers:
            if verbose:
                print('start training %s classifier' % cls)
            # check if there is any training data at all
            X, y = training_data[cls]
            if len(X) == 0:
                if verbose:
                    print('no training data available.')
                continue
            # if so, convert it into numpy format
            X = np.stack(X, 0)
            y = np.array(y)
            # check if there is there are at least two unique
            # labels
            y_uniq = np.unique(y)
            # if not, use a constant classifier.
            if len(y_uniq) == 1:
                if verbose:
                    print('using a constant classifier with value %s' % str(y_uniq[0]))
                setattr(self, cls, int(y_uniq[0]))
                continue
            # check if there is too much training data
            if self.max_points is not None and X.shape[0] > self.max_points:
                if verbose:
                    print('too much trianing data (%d points); subsampling to %d points' % (X.shape[0], self.max_points))
                subset = np.random.choice(len(X), self.max_points, False)
                X = X[subset, :]
                y = y[subset]
            # initialize an SVM and train it
            svm = SVC(C = self.C)
            svm.fit(X, y)
            if verbose:
                print('training accuracy: %g' % svm.score(X, y))
            setattr(self, cls, svm)
