"""
A mechanism to predict which variables are used at which point in a decoded
tree.

"""

# Copyright (C) 2020
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
__version__ = '0.1.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'benjamin.paassen@sydney.edu.au'

from tqdm import tqdm
import torch
import random
import tree
import numpy as np
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin
import python_ast_utils

class VariableClassifier(BaseEstimator):
    """ Implements a classifier to decide which variable references should be
    used in a given abstract syntax tree. The internally used classifier is a
    RBF kernel support vector machine.

    This mechanism relies on a recursive autoencoder to be provided.

    In more detail, this model auto-encodes an input program using teacher
    forcing and uses the vectors produced during decoding to decide on the
    variable reference whenever necessary. Variable references are deemed
    necessary whenever a 'Name' node is encountered in the syntax tree.
    A variable reference can either be selected from a list of global variables
    (like 'print') or a local scope. A new variable is put into the local scope
    whenever a new class, function, or variable is defined (via Assign), or
    whenever an 'arg' is encountered (like function arguments).

    Generally, the internal classifier has as many classes as global and local
    variables occur.

    Parameters
    ----------
    autoencoder: class recursive_tree_grammar_auto_encoder.TreeGrammarAutoEncoder
        A pre-trained autoencoder for computer programs.
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

    def fit(self, trees, verbose = False, return_codes = False):
        # accumulate training data
        X_vars = []
        Y_vars = []
        X_funs = []
        Y_funs = []
        # also maintain a dictionary of global variables across programs
        global_vars = {}

        if verbose:
            print('Start pre-processing the trees.')
            progbar = tqdm
        else:
            progbar = lambda x : x

        if return_codes:
            codes = []

        # iterate over all trees
        for j in progbar(range(len(trees))):
            nodes, adj, var = trees[j][0], trees[j][1], trees[j][2]
            # pre-compute the left-hand-side of assignments because these
            # can introduce new local variables
            assign_lefts = assign_lefts_(nodes, adj)
            # pre-compute function calls because we distinguish between
            # variables and callables
            calls = calls_(nodes, adj)

            # autoencode the tree to obtain the vectors corresponding
            # to each production step
            seq, h, nonts, H = self.subtree_encodings_(nodes, adj)
            if return_codes:
                codes.append(h)
            # set up a dictionary of local variables in this program
            local_vars = {}
            # set up a dictionary of local functions/classes in this program
            local_funs = {}
            # move through the tree via depth-first-search and search for
            # points where a variable classifier would have to decide for a
            # variable or a function. In both cases, we append a training
            # data point
            i = 0
            for t in range(len(seq)):
                A = nonts[t]
                r = seq[t]
                h = H[t]
                # check the current nonterminal
                if isinstance(A, str) and (A.endswith('?') or A.endswith('*')):
                    continue
                # check the produced terminal symbol and validate against node
                # list
                if nodes[i] != self.autoencoder._grammar._rules[A][r][0]:
                    raise ValueError('Expected %s as next node but got %d' % (nodes[i], self.autoencoder._grammar._rules[A][r][0]))
                if i in var:
                    # check if we initialize a new local variable in this node
                    k = -1
                    for new_name_attr in ['arg', 'name']:
                        if new_name_attr in var[i]:
                            name = var[i][new_name_attr]
                            if nodes[i] in ['ClassDef', 'FunctionDef']:
                                if name not in local_funs:
                                    k = len(python_ast_utils._builtin_funs) + len(local_funs)
                                    local_funs[name] = k
                            elif name not in local_vars:
                                k = len(local_vars) + 1
                                local_vars[name] = k
                    # check if a variable/function is referenced
                    if 'id' in var[i]:
                        name = var[i]['id']
                        k = -1
                        if i in calls:
                            # if this is a function call, check if
                            # it references a local function
                            if name in local_funs:
                                k = local_funs[name]
                            elif name in python_ast_utils._builtin_funs:
                                # otherwise it needs to be a global function
                                k = python_ast_utils._builtin_funs.index(name)
                            else:
                                # or we ignore it because it references
                                # something that doesn't exist
                                pass
                            if k >= 0:
                                X_funs.append(h)
                                Y_funs.append(k)
                        else:
                            # check if we reference a local variable
                            if name in local_vars:
                                k = local_vars[name]
                            elif i in assign_lefts:
                                # if it is not a local variable but is the left-hand-side
                                # of an assignment, create a new local variable
                                k = len(local_vars) + 1
                                local_vars[name] = k
                                k = 0
                            else:
                                # otherwise we ignore it because it references
                                # something that doesn't exist
                                pass
                            if k >= 0:
                                X_vars.append(h)
                                Y_vars.append(k)
                i += 1

        if len(X_vars) > 0:

            if verbose:
                print('Start training variable classifier.')
            # accumulate the training data across trees
            X_vars = torch.stack(X_vars).detach().numpy()
            Y_vars = np.array(Y_vars)

            if self.max_points is not None and len(X_vars) > self.max_points:
                if verbose:
                    print('Too much training data (%d points); subsampling to %d points.' % (len(X_vars), self.max_points))
                subset = np.random.choice(len(X_vars), self.max_points, False)
                X_vars = X_vars[subset, :]
                Y_vars = Y_vars[subset]

            # train the variable classification SVM
            if len(np.unique(Y_vars)) > 1:
                self.c_vars_ = SVC(C = self.C)
                self.c_vars_.fit(X_vars, Y_vars)
            else:
                self.c_vars_ = ConstantClassifier()
                self.c_vars_.fit(X_vars, Y_vars)

            if verbose:
                print('training accuracy: %g' % self.c_vars_.score(X_vars, Y_vars))

        if len(X_funs) > 0:
            if verbose:
                print('Start training function classifier.')


            # accumulate the training data across trees
            X_funs = torch.stack(X_funs).detach().numpy()
            Y_funs = np.array(Y_funs)

            if self.max_points is not None and len(X_funs) > self.max_points:
                if verbose:
                    print('Too much training data (%d points); subsampling to %d points.' % (len(X_funs), self.max_points))
                subset = np.random.choice(len(X_funs), self.max_points, False)
                X_funs = X_funs[subset, :]
                Y_funs = Y_funs[subset]

            # train the function classification SVM
            if len(np.unique(Y_funs)) > 1:
                self.c_funs_ = SVC(C = self.C)
                self.c_funs_.fit(X_funs, Y_funs)
            else:
                self.c_funs_ = ConstantClassifier()
                self.c_funs_.fit(X_funs, Y_funs)

            if verbose:
                print('training accuracy: %g' % self.c_funs_.score(X_funs, Y_funs))

        if return_codes:
            return self, torch.stack(codes)
        else:
            return self

    def predict(self, nodes, adj, verbose = False):
        """ Predicts the variable dictionary for the given tree. """

        # pre-compute the left-hand-side of assignments because these
        # can introduce new local variables
        assign_lefts = assign_lefts_(nodes, adj)
        # pre-compute function calls because we distinguish between
        # variables and callables
        calls = calls_(nodes, adj)

        if verbose:
            print('input tree: %s' % tree.tree_to_string(nodes, adj, indent = True, with_indices = True))
            print('assign lefts: %s' % str(assign_lefts))
            print('calls: %s' % str(calls))

        # autoencode the tree to get encoding vectors
        seq, _, nonts, H = self.subtree_encodings_(nodes, adj)
        # initialize the var dictionary
        var = {}
        # initialize lists of local variables and local functions
        local_vars = []
        local_funs = []
        # start iterating over the tree
        i = 0
        for t in range(len(seq)):
            A = nonts[t]
            r = seq[t]
            # check the current nonterminal
            if isinstance(A, str) and (A.endswith('?') or A.endswith('*')):
                continue
            # check the produced terminal symbol and validate against node
            # list
            if nodes[i] != self.autoencoder._grammar._rules[A][r][0]:
                raise ValueError('Expected %s as next node but got %d' % (nodes[i], self.autoencoder._grammar._rules[A][r][0]))
            # create a local function for class definitions and function
            # definitions and create a local variable for arguments
            if nodes[i] in ['ClassDef', 'FunctionDef']:
                name = 'callable%d' % len(local_funs)
                var[i] = {'name' : name}
                local_vars.append(name)
            elif nodes[i] == 'arg':
                name = 'var%d' % len(local_vars)
                var[i] = {'arg' : name}
                local_vars.append(name)
            elif nodes[i] in ['Name']:
                h = H[t].detach().numpy()
                # if we encounter a reference to a variable, check first if we
                # are in a function call
                if i in calls:
                    if not hasattr(self, 'c_funs_'):
                        # if we don't have a function classifier, we need to
                        # set an erroneous name
                        name = '<some_function>'
                    else:
                        # if so, ask the funtion classifier which function we
                        # should reference
                        y = sorted_decision_(self.c_funs_, h)
                        # but only accept an answer in the possible range, i.e.
                        # we can not select a local function that's not there
                        k = np.argmax(y[:len(python_ast_utils._builtin_funs) + len(local_funs)])
                        # check if we selected a local function
                        if k >= len(python_ast_utils._builtin_funs):
                            name = local_funs[k - len(python_ast_utils._builtin_funs)]
                        else:
                            # if not we use a builtin function
                            name = python_ast_utils._builtin_funs[k]
                else:
                    if not hasattr(self, 'c_vars_'):
                        # if we don't have a variable classifier, we need to
                        # set an erroneous name
                        name = '<some_variable>'
                    else:
                        # otherwise we need to select a local variable
                        y = sorted_decision_(self.c_vars_, h)
                        # but only accept an answer in the possible range, i.e.
                        # we can not select a local variable that's not there
                        k = np.argmax(y[:len(local_vars) + 1])
                        if k > 0:
                            # if k is larger zero, select the corresponding local
                            # variable and we are finished
                            name = local_vars[k-1]
                        else:
                            # otherwise, one of three things may occur
                            if i in assign_lefts:
                                # if we are at the left side of an assignment,
                                # we introduce a new local variable
                                name = 'var%d' % len(local_vars)
                                local_vars.append(name)
                            elif len(local_vars) > 0:
                                # otherwise we need to forbid k = 0 and choose the
                                # next-best local variable
                                k = 1 + np.argmax(y[1:len(local_vars)+1])
                                name = local_vars[k-1]
                            else:
                                # if no local variable is available, we are at an
                                # impasse and just add an error variable
                                name = '<some_variable>'
                var[i] = {'id' : name}
            i += 1
        return var

    def subtree_encodings_(self, nodes, adj):
        """ Returns the rule sequence and all subtree decodings for the given
        tree.

        """
        seq, z = self.autoencoder.encode(nodes, adj)
        h = self.autoencoder._nonlin(self.autoencoder._vae_dec(z))
        _, _, nonts, H = self.autoencoder._dec.produce(h, seq)
        return seq, z, nonts, H

def sorted_decision_(svm, x):
    """ Computes the SVM decision function for the given input
    vector and reorders it such that the decision value for class
    k is in entry y[k].

    Parameters
    ----------
    svm: class sklearn.svm.SVC
        a support vector machine.
    x: ndarray of shape (n,)
        A single vector to be classified.

    Returns
    -------
    y: ndarray of shape (max(svm.classes_)+1,)
        The decision function values for the input vector.

    """
    yraw = svm.decision_function(np.expand_dims(x, 0))
    y = np.full(np.max(svm.classes_)+1, -np.inf)
    if len(svm.classes_) == 2:
        y[svm.classes_[0]] = 0.
        y[svm.classes_[1]] = yraw[0]
    else:
        y[svm.classes_] = yraw
    return y

def assign_lefts_(nodes, adj):
    """ Returns a set of node indices that are the left-hand-side of
    assignments.

    """
    assign_lefts = set()
    for i in range(len(nodes)):
        if nodes[i] in ['For', 'Assign', 'AugAssign', 'AnnAssign']:
            stk = [adj[i][0]]
            while stk:
                j = stk.pop()
                if nodes[j] == 'Name':
                    assign_lefts.add(j)
                for l in adj[j]:
                    stk.append(l)
    return assign_lefts

def calls_(nodes, adj):
    """ Returns a set of node indices that need to be callable functions.

    """
    calls = set()
    for i in range(len(nodes)):
        if nodes[i] == 'Call':
            stk = [adj[i][0]]
            while stk:
                j = stk.pop()
                if nodes[j] == 'Name' or nodes[j] == 'Attribute':
                    # break as soon as we find the outermost right name
                    calls.add(j)
                    break
                for l in adj[j]:
                    stk.append(l)
    return calls

class ConstantClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self):
        pass

    def fit(self, X, Y):
        unique_classes = np.unique(Y)
        if len(unique_classes) > 1:
            raise ValueError('Can not fit a constant classifier with more than one class.')
        self.classes_ = unique_classes

    def predict(self, X):
        N, n = X.shape

        return np.full(N, self.classes_[0])

    def decision_function(self, X):
        N, n = X.shape

        return np.full((N, 1), 1.)
