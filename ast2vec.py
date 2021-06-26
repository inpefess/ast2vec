"""
Provides a convenient interface to load the ast2vec model and perform typical
operations with it.

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

import torch
import recursive_tree_grammar_auto_encoder as rtg_ae
import python_ast_utils
import astor
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

DIM_     = 256
DIM_VAE_ = 256

def load_model(path = 'ast2vec.pt'):
    """ Loads the ast2vec model from the given path.

    Parameters
    ----------
    path: str
        The path to the ast2vec model file, which should be called 'ast2vec.pt'.

    Returns
    -------
    class recursive_tree_grammar_auto_encoder.TreeGrammarAutoEncoder
        An instance of the ast2vec model.

    """
    # initialize a new autoencoder instance
    model         = rtg_ae.TreeGrammarAutoEncoder(python_ast_utils.grammar, dim = DIM_, dim_vae = DIM_VAE_)
    # load the model parameters into it
    model.load_state_dict(torch.load(path))

    # return the model
    return model

def encode_trees(model, trees, verbose = False):
    """ Encodes the given trees and returns the point encodings as a numpy
    matrix.

    Paramters
    ---------
    model: class recursive_tree_grammar_auto_encoder.TreeGrammarAutoEncoder
        An instance of the ast2vec model.
    trees: list
        A list of trees, each a tuple of node list and adjacency list.
    verbose: bool (default = False)
        If set to true, we show a progress bar.

    Returns
    -------
    X: ndarray of size len(list) x DIM_VAE_
        The point encodings for all input trees.

    """
    # set up the progress bar if so desired
    if verbose:
        progbar = tqdm
    else:
        progbar = lambda x : x

    # initialize the encoding matrix
    X = np.zeros((len(trees), DIM_VAE_))
    # iterate over all trees
    for i in progbar(range(len(trees))):
        nodes = trees[i][0]
        adj   = trees[i][1]
        # encode the current tree
        _, x = model.encode(nodes, adj)
        # convert the vector to numpy and store it in the matrix
        X[i, :] = x.detach().numpy()

    # return the matrix
    return X

def decode_points(model, X, max_size = 100, verbose = False):
    """ Decodes a given matrix of points to trees.

    Paramters
    ---------
    model: class recursive_tree_grammar_auto_encoder.TreeGrammarAutoEncoder
        An instance of the ast2vec model.
    X: ndarray of size m x DIM_VAE_
        A matrix of points to be decoded.
    max_size: int (default = 100)
        A maximum tree size for decoding to prevent endless loops.
    verbose: bool (default = False)
        If set to true, we show a progress bar.

    Returns
    -------
    trees: list
        A list of decoded trees, each a tuple of node list and adjacency list.

    """
    # set up the progress bar if so desired
    if verbose:
        progbar = tqdm
    else:
        progbar = lambda x : x

    # initialize the tree list
    trees = []
    # iterate over all points
    for i in progbar(range(X.shape[0])):
        # convert point to torch tensor
        x = torch.tensor(X[i, :], dtype=torch.float)
        # decode the point
        nodes, adj, _ = model.decode(x, max_size = max_size)
        trees.append((nodes, adj))
    # return the tree list
    return trees


def progress_pca(x, y, X):
    """ Performs a variant of 2D principal component analysis which selects
    one axis as the direction vector from start to target and the other
    axis as orthogonal such that it maximizes the remaining variance.

    In more detail, we project the data into the orthogonal space to
    the direction vector from start to target and then perform a standard
    PCA.

    This function returns an orthonormal matrix W, a scale factor, and
    a bias vector x, such that the low-dimensional representation of the data
    is obtained as

    Y = np.dot(W, (X - x)) / scale

    The inverse projection from 2D into the high dimensional space is obtained
    as

    X = np.dot(W.T, Y * scale) + x

    Note that the special way we set up our construction ensures that the
    start vector corresponds exactly to the origin and the target vector
    corresponds exactly to point (1, 0) in our low-dim coordinate system.
    In other words, for these two special points, our PCA variant is exact
    with no reconstruction error (up to numerics). More precisely, we obtain

    x = np.dot(W.T, 0 * scale) + x and
    y = np.dot(W.T, [1, 0] * scale) + x

    The first equation is trivial. The second becomes obvious once it is
    clear that W[0, :] * scale is equal to y - x.

    Parameters
    ----------
    x: ndarray of size DIM_VAE_ OR int
        The start vector OR index of the start vector in the X matrix.
    y: ndarray of size DIM_VAE_ OR int
        The target vector OR index of the target vector in the X matrix.
    X: ndarray of size m x DIM_VAE_
        The data matrix.

    Returns
    -------
    W: ndarray of size 2 x DIM_VAE_
        The orthonormal projection matrix from high to low-dim space.
    scale: float
        The scale parameter, i.e. the Euclidean distance between x and y.
    x: ndarray of size DIM_VAE_
        The same x as in the input.

    """
    # preprocess the input
    if isinstance(x, int):
        x = X[x, :]
    if isinstance(y, int):
        y = X[y, :]

    # get the difference vector
    delta = y - x
    scale   = np.linalg.norm(delta)
    delta   = np.expand_dims(delta / scale, 0)

    # project delta out of the data to get the orthogonal subspace
    Xsubsp  = X - x
    Xproj   = np.dot(np.dot(Xsubsp, delta.T), delta)
    Xsubsp -= Xproj

    # perform a PCA on the orthogonal subspace. In other words, we
    # compute the eigenvalue decomposition of the covariance matrix
    # of the orthogonal subspace, take the eigenvector corresponding
    # to the largest eigenvalue, and use that as our second axis.
    Pseudocov = np.dot(Xsubsp.T, Xsubsp)
    eigenvalues, V = np.linalg.eig(Pseudocov)
    V = np.real(V)
    k = np.argmax(np.real(eigenvalues))
    # take largest component and build projection matrix
    W = np.concatenate((delta, np.expand_dims(V[:, k], 0)), 0)

    # return result
    return W, scale, x


def interpolation_grid(model, start_tree, target_tree, grid_size = 11, X = None, max_size = 100):
    """ Samples a grid of evenly spaces points between
    the start tree and the end tree in the coding space, decodes the points
    back into trees and returns the grid in coding space, an index grid of
    which point mapped to which tree, and a list of decoded trees.

    If a reference data set of encoded trees is given, the grid is not only
    one-dimensional but two-dimensional. The x-axis still corresponds to the
    direct line between start and end encoding, but the y-axis then corresponds
    to the orthogonal axis of highest variance in the data set, equivalent to
    principal component analysis.

    Parameters
    ----------
    model: class recursive_tree_grammar_auto_encoder.TreeGrammarAutoEncoder
        An instance of the ast2vec model.
    start_tree: tuple OR int
        The tree where we start the interpolation, represented as a tuple of
        node list and adjacency list. If the X argument is given, this can also
        be just the index of the start data point.
    target_tree: tuple OR int
        The tree where we end the interpolation, represented as a tuple of
        node list and adjacency list. If the X argument is given, this can also
        be just the index of the target data point.
    grid_size: int (default = 11)
        The number of regular samples between start and target.
    X: ndarray of size m x DIM_VAE_ (default = None)
        A matrix of reference encodings
    max_size: int (default = 100)
        A maximum tree size for decoding to prevent endless loops.

    Returns
    -------
    points: ndarray of size grid_size x DIM_VAE_
        A matrix containing the regular samples between start and target. The
        first sample is the encoding of start, the last of target. If the X
        argument is given, this is a tensor of size
        grid_size x grid_size x DIM_VAE_ where the start is at location
        points[0, grid_size // 2, :] in the tensor and the target is at
        location points[-1, grid_size // 2, :] in the tensor.
    grid: ndarray of size grid_size
        A vector of integers indexing the decoded trees. In particular, we
        get the tree with index grid[i] when we decode points[i, :]. If the
        X argument is given, this is a matrix of size grid_size x grid_size.
    trees: list
        A list of decoded trees, such that trees[grid[i]] is the tree when
        we decode points[i, :].

    """
    # consider first the case without data set argument
    if X is None:
        # encode start and target tree
        _, x    = model.encode(start_tree[0], start_tree[1])
        x       = x.detach().numpy()

        _, y    = model.encode(target_tree[0], target_tree[1])
        y       = y.detach().numpy()

        # initialize output
        points = np.zeros((grid_size, DIM_VAE_))
        grid   = np.zeros(grid_size, dtype=int)
        trees  = []

        # interpolate linearly between them
        for i in range(grid_size):
            # compute interpolated point
            alpha = i / (grid_size-1)
            points[i, :] = (1. - alpha) * x + alpha * y
            # decode it
            nodes, adj, _ = model.decode(torch.tensor(points[i, :], dtype=torch.float), max_size = max_size)
            # check if this tree already occured
            j = 0
            while j < len(trees):
                if nodes == trees[j][0] and adj == trees[j][1]:
                    break
                j += 1
            # store the tree index in the grid
            grid[i] = j
            # the the index didn't occur before, append the new tree to the
            # list
            if j == len(trees):
                trees.append((nodes, adj))
        # return result
        return points, grid, trees
    else:
        # consider the case with data set argument

        # encode start and target tree if necessary
        if not isinstance(start_tree, int):
            _, x    = model.encode(start_tree[0], start_tree[1])
            x       = x.detach().numpy()
        else:
            x       = X[start_tree, :]

        if not isinstance(target_tree, int):
            _, y    = model.encode(target_tree[0], target_tree[1])
            y       = y.detach().numpy()
        else:
            y       = X[target_tree, :]


        # project the data down to 2D using a special kind of PCA that
        # preserves the direction vector from start to target
        W, scale, _ = progress_pca(x, y, X)

        # sample a regular grid in low-dim space
        Ygrid   = []
        for i in np.linspace(0, 1, grid_size):
            for j in np.linspace(-0.5, +0.5, grid_size):
                Ygrid.append([i, j])
        Ygrid  = np.array(Ygrid)
        # project into high-dim space via inverse PCA
        Xgrid   = np.dot(Ygrid, W) * scale + x

        # initialize output
        points  = np.zeros((grid_size, grid_size, DIM_VAE_))
        grid  = np.zeros((grid_size, grid_size), dtype=int)
        trees = []

        # decode all points in the regular grid
        k = 0
        for i in range(grid_size):
            for j in range(grid_size):
                # retrieve currrent point
                points[i, j, :] = Xgrid[k, :]
                k += 1
                # decode it
                nodes, adj, _ = model.decode(torch.tensor(points[i, j, :], dtype=torch.float), max_size = max_size)
                # check if this tree already occured
                l = 0
                while l < len(trees):
                    if nodes == trees[l][0] and adj == trees[l][1]:
                        break
                    l += 1
                # store the tree index in the grid
                grid[i, j] = l
                # the the index didn't occur before, append the new tree to the
                # list
                if l == len(trees):
                    trees.append((nodes, adj))

        # return the output
        return points, grid, trees

def interpolation_plot(model, start_tree, target_tree, grid_size = 11, X = None, max_size = 100, variable_classifier = None, plot_code = 3):
    """ Does an interpolation plot, i.e. plots a grid between the start and the
    end tree, coloring the grid according to the tree the grid point decodes
    to. 

    If the optional X argument is given, this is a 2D grid, otherwise it is
    only 1D.

    The plot_code most common trees are printed into the plot.

    Parameters
    ----------
    model: class recursive_tree_grammar_auto_encoder.TreeGrammarAutoEncoder
        An instance of the ast2vec model.
    start_tree: tuple OR int
        The tree where we start the interpolation, represented as a tuple of
        node list and adjacency list. If the X argument is given, this can also
        be just the index of the start data point.
    target_tree: tuple OR int
        The tree where we end the interpolation, represented as a tuple of
        node list and adjacency list. If the X argument is given, this can also
        be just the index of the target data point.
    grid_size: int (default = 11)
        The number of regular samples between start and target.
    X: ndarray of size m x DIM_VAE_ (default = None)
        A matrix of reference encodings
    max_size: int (default = 100)
        A maximum tree size for decoding to prevent endless loops.
    variable_classifier: class variable_classifier.VariableClassifier (default = None)
        An optional variable classifier to make the printed code more
        clear.
    plot_code: int (default = 3)
        The source code corresponding to the most frequent plot_code trees in
        the grid is included in the plot.

    """
    # do the interpolation
    _, grid, grid_trees = interpolation_grid(model, start_tree, target_tree, grid_size = grid_size, X = X, max_size = max_size)

    # count the most frequent trees
    histogram = np.zeros(len(grid_trees), dtype=int)
    for k in range(len(grid_trees)):
        histogram[k] = np.sum(grid == k)

    if X is None:
        grid = np.expand_dims(grid, 1)

    # plot the grid
    offset = 0.5 / (grid_size-1)
    plt.imshow(grid, origin = 'lower', extent = [0. - offset, 1. + offset, -0.5 - offset, +0.5 + offset])

    # plot the source code for the most frequent trees
    if plot_code is not None and plot_code > 0:
        freq_trees = np.argsort(histogram)[len(grid_trees)-plot_code:]

        xs = np.linspace(0, 1, grid_size)
        ys = np.linspace(-0.5, +0.5, grid_size)

        for k in freq_trees:
            mean    = np.zeros(2)
            counter = 0
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    if grid[i, j] != k:
                        continue
                    mean[0] += xs[i]
                    mean[1] += ys[j]
                    counter += 1
            mean /= counter
            if X is None:
                mean[1] = 0.

            nodes, adj = grid_trees[k]
            if variable_classifier is not None:
                var = variable_classifier.predict(nodes, adj)
            else:
                var = None
            ast = python_ast_utils.tree_to_ast(nodes, adj, var)
            src = astor.to_source(ast)
            if src == '':
                src = '<empty program>'
            plt.text(mean[0], mean[1], src, bbox=dict(facecolor='white', alpha=0.7))


def traces_plot(start, target, traces, X, trees = None, plot_code = 0):
    """ Plots the movement of points through the latent space.

    To map the data to 2D, we use the progress_pca function from above.
    In the 2D space, we then draw a quiver plot.

    Parameters
    ----------
    start: int
        The index of the start tree for all students.
    target: int
        The inex of the target tree for all students, e.g. the index of the
        most popular correct solution.
    traces: list
        A list of student traces in terms of indices, i.e. a list of lists,
        where each entry is the index of a tree.
    X: ndarray of size m x DIM_VAE_
        The data matrix such that X[start, :] is the point representation
        of the initial student state, X[target, :] represents the correct
        solution, and X[traces[i][t], :] represents the t-th step of the i-th
        student trace in the data.
    trees: list (default = None)
        A list of trees as triples of node list, adjacency list, and variable
        maps from the student data. This is necessary to draw example student
        code. Otherwise, only the arrows are drawn.
    plot_code: int (default = 0)
        The source code corresponding to the most frequent plot_code trees in
        the grid is included in the plot.

    """
    # get start and target vector
    x = X[start, :]
    y = X[target, :]

    # map the data down to 2D via progress pca
    W, scale, _ = progress_pca(x, y, X)
    Xlo = np.dot(X - x, W.T) / scale

    # build a histogram of how often trees occur in traces
    histogram = np.zeros(len(X), dtype=int)
    for i in range(len(traces)):
        for t in range(len(traces[i])):
            histogram[traces[i][t]] += 1
    # check if the start is explicitly included in traces. If not, we add it
    # artificially
    start_included = np.all([trace[0] == start for trace in traces])
    if not start_included:
        histogram[start] += len(traces)

    # accumulate arrows
    Xarr = []
    Varr = []
    Carr = []
    for k in range(len(traces)):
        trace = traces[k]
        if not start_included:
            trace = [start] + trace
        for t in range(1, len(trace)):
            i = trace[t-1]
            j = trace[t]
            Xarr.append(Xlo[i, :])
            Varr.append(Xlo[j, :] - Xlo[i, :])
            Carr.append(k)
    Xarr = np.stack(Xarr, 0)
    Varr = np.stack(Varr, 0)

    # draw all points that occur at least once
    plt.scatter(Xlo[histogram >= 1, 0], Xlo[histogram >= 1, 1], c = histogram[histogram >= 1], s = 50)

    # draw all arrows
    plt.quiver(Xarr[:, 0], Xarr[:, 1], Varr[:, 0], Varr[:, 1], Carr, angles='xy', scale_units='xy', scale=1., width = 0.005)

    if plot_code > 0:
        text_offset =  0.05 * (np.max(Xlo[:, 0]) - np.min(Xlo[:, 0]))

        # plot the most common programs
        freq_trees = np.argsort(histogram)[len(X)-plot_code:]

        for k in freq_trees:
            nodes, adj, var = trees[k]
            ast = python_ast_utils.tree_to_ast(nodes, adj, var)
            src = astor.to_source(ast)
            if src == '':
                src = '<empty program>'
            plt.text(Xlo[k, 0] + text_offset, Xlo[k, 1], src, bbox=dict(facecolor='white', alpha=0.7), horizontalalignment = 'left', verticalalignment = 'bottom')

def construct_dynamical_system(y, X, traces, regul = 1E-3):
    """ Constructs a linear dynamical system that has a fix point at y and
    otherwise roughly follows the student movement in traces.

    In particular, the dynamical system has the form

    f(z) = z + W * (y - z)

    which obviously has a fix point at z = y.

    W is learned via a linear regression such that f(X[traces[i][t]]) is as
    close as possible to X[traces[i][t+1]] for all i and t.
    Provided that the regularization parameter of the linear regression is high
    enough, it can be shown that y is the unique attractor of the dynamical
    system.

    Parameters
    ----------
    y: ndarray of size DIM_VAE_ OR int
        The target vector OR index of the target vector in the X matrix.
    X: ndarray of size m x DIM_VAE_
        The data matrix.
    traces: list
        A list of student traces in terms of indices, i.e. a list of lists,
        where each entry is the index of a tree.
    regul: float (default = 1E-3)
        Some positive real number indicating the regularization strength. For
        high values, W becomes close to the identity matrix scaled with some
        small number.

    Returns
    -------
    W: ndarray of size DIM_VAE_ x DIM_VAE_
        The matrix governing the dynamical system.

    """
    # preprocess the input
    if isinstance(y, int):
        y = X[y, :]

    # prepare the training data for the linear regression
    Xdyn  = []
    Ydyn  = []
    for trace in traces:
        for t in range(1, len(trace)):
            i = trace[t-1]
            j = trace[t]
            Xdyn.append(X[i, :])
            Ydyn.append(X[j, :])
    Xdyn  = np.stack(Xdyn, 0)
    Ydyn  = np.stack(Ydyn, 0)

    # to adjust the training data we set the outputs to
    Ydyn  = Ydyn - Xdyn
    # and the inputs to
    Xdyn  = y - Xdyn
    # perform the linear regression
    W = np.linalg.solve(np.dot(Xdyn.T, Xdyn) + regul * np.eye(Xdyn.shape[1]), np.dot(Xdyn.T, Ydyn))

    # return the result
    return W


def dynamical_system_plot(W, start_tree, target_tree, X, grid_size = 11, arrow_scale = 10., step_size = 1., max_steps = 10, model = None, max_size = 100, variable_classifier = None):
    """ Visualizes a linear dynamical system via a regular grid in two
    dimensions.

    We first do an interpolation_plot and then draw the dynamical system on
    top. The dynamical system is assumed to be of the form

    f(z) = z + W * (y - z)

    We visualize the predictions via a quiver plot.

    Parameters
    ----------
    W: ndarray of size DIM_VAE_ x DIM_VAE_
        matrix describing the linear dynamical system.
    start_tree: tuple OR int
        The tree where we start the interpolation, represented as a tuple of
        node list and adjacency list. If the X argument is given, this can also
        be just the index of the start data point.
    target_tree: tuple OR int
        The tree where we end the interpolation, represented as a tuple of
        node list and adjacency list. If the X argument is given, this can also
        be just the index of the target data point.
    X: ndarray of size m x DIM_VAE_
        A matrix of reference encodings
    grid_size: int (default = 11)
        The number of regular samples between start and target.
    arrow_scale: float (default = 10.)
        The arrow scaling for the dynamical systems plot. Higher values make
        the arrows shorter, which can be beneficial to make the image easier
        to parse.
    step_size: float (default = 1.)
        If a positive number is given, a trace will be simulated that starts
        at start_tree and then follows the dynamical system using Euler's
        method with the given step size. The trace concludes if it is very
        close to the target_tree or after max_steps.
    max_steps: int (default = 10)
        The maximum number of steps in the simulated trace.
    model: class recursive_tree_grammar_auto_encoder.TreeGrammarAutoEncoder (default = None)
        An instance of the ast2vec model. If given, the trees in the simulated
        trace will be decoded as well.
    max_size: int (default = 100)
        A maximum tree size for decoding to prevent endless loops.
    variable_classifier: class variable_classifier.VariableClassifier (default = None)
        An optional variable classifier to make the printed code more clear.

    """
    # do the interpolation
    points, grid, grid_trees = interpolation_grid(model, start_tree, target_tree, grid_size = grid_size, X = X, max_size = max_size)

    # compute the dynamical system predictions for all points in the grid
    x = points[0, grid_size // 2, :]
    y = points[-1, grid_size // 2, :]
    Xgrid = points.reshape((grid_size ** 2, DIM_VAE_))
    Ugrid = np.dot(y - Xgrid, W)

    # map them back to the low-dim representation
    V, scale, _ = progress_pca(x, y, X)
    Xgridlo = np.dot(Xgrid - x, V.T) / scale
    Ugridlo = np.dot(Ugrid, V.T) / scale

    # plot the grid
    offset = 0.5 / (grid_size-1)
    plt.imshow(grid, origin = 'lower', extent = [0. - offset, 1. + offset, -0.5 - offset, +0.5 + offset])

    # plot the dynamical system as a quiver plot
    plt.quiver(Xgridlo[:, 0], Xgridlo[:, 1], Ugridlo[:, 0], Ugridlo[:, 1], angles='xy', scale_units='xy', scale=arrow_scale, width = 0.005, color = 'red')

    # if so desired, simulate a trace
    if step_size > 0.:
        Xtrace = []
        Vtrace = []
        xtrace = x

        for t in range(max_steps):
            # make one Euler step
            vtrace = np.dot(W.T, y - xtrace) * step_size
            # append it
            Xtrace.append(xtrace)
            Vtrace.append(vtrace)
            # update the point
            xtrace = xtrace + vtrace
            # if we have converged to y, stop
            if np.sum(np.square(xtrace - y)) < 1E-1:
                break

        Xtrace = np.stack(Xtrace)
        Vtrace = np.stack(Vtrace)

        # map to low dim
        Xtracelo = np.dot(Xtrace - x, V.T) / scale
        Vtracelo = np.dot(Vtrace, V.T) / scale

        # plot
        plt.quiver(Xtracelo[:, 0], Xtracelo[:, 1], Vtracelo[:, 0], Vtracelo[:, 1], angles='xy', scale_units='xy', scale=1., width = 0.005, color = 'blue')

        # plot the source code for the steps of the trace
        if model is not None:
            text_offset =  0.05 * (np.max(Xgridlo[:, 1]) - np.min(Xgridlo[:, 1]))
            trees_trace = decode_points(model, Xtrace, max_size = max_size)
            last_nodes = None
            last_adj   = None
            for t in range(Xtrace.shape[0]):
                nodes, adj = trees_trace[t]
                if nodes == last_nodes and adj == last_adj:
                    continue
                last_nodes = nodes
                last_adj   = adj
                if variable_classifier is not None:
                    var = variable_classifier.predict(nodes, adj)
                else:
                    var = None
                ast = python_ast_utils.tree_to_ast(nodes, adj, var)
                src = astor.to_source(ast)
                if src == '':
                    src = '<empty program>'
                plt.text(Xtracelo[t, 0], Xtracelo[t, 1] + text_offset, src, bbox=dict(facecolor='white', alpha=0.7), verticalalignment='bottom')


def cluster_plot(start, target, X, Y, means = None, traces = None, model = None, max_size = 100, variable_classifier = None):
    """ Plots a clustering of the data X in two dimensions, using
    progress_pca for dimensionality reduction. In particular, points are
    plotted in color according to their cluster label. Optionally, this
    function can plot the traces between points, the cluster means, and decode
    the cluster means into programs.

    Parameters
    ----------
    start: int
        The index of the start tree for all students.
    target: int
        The inex of the target tree for all students, e.g. the index of the
        most popular correct solution.
    X: ndarray of size m x DIM_VAE_
        The data matrix such that X[start, :] is the point representation
        of the initial student state, and X[target, :] represents the correct
        solution.
    Y: ndarray of size m
        The cluster labels for each datapoint.
    means: ndarray of size K x DIM_VAE_ (default = None)
        The cluster means. If given, these are plotted as well. Note that the
        array is assumed to be in order of the indices in np.unique(Y).
    traces: list (default = None)
        A list of student traces in terms of indices, i.e. a list of lists,
        where each entry is the index of a tree. X[traces[i][t], :] is assumed
        to be the location of the t-th step of the i-th trace. If given, the
        traces are shown as arrows.
    model: class recursive_tree_grammar_auto_encoder.TreeGrammarAutoEncoder (default = None)
        An instance of the ast2vec model. If given, cluster means are decoded
        into programs.
    max_size: int (default = 100)
        Only relevant if means and model are given. In that case, this is the
        maximum tree size for decoding a program.
    variable_classifier: class variable_classifier.VariableClassifier (default = None)
        Only relevant if means and model are given. An optional variable
        classifier to make the printed code more clear.

    """

    # get start and target vector
    x = X[start, :]
    y = X[target, :]

    # map the data down to 2D via progress pca
    W, scale, _ = progress_pca(x, y, X)
    Xlo = np.dot(X - x, W.T) / scale

    # plot the points in colors depending on their cluster assignment
    plt.scatter(Xlo[:, 0], Xlo[:, 1], c = Y, s = 50)

    # plot the traces if given
    if traces is not None:
        # check if the start is explicitly included in traces. If not, we add it
        # artificially
        start_included = np.all([trace[0] == start for trace in traces])
        if not start_included:
            histogram[start] += len(traces)

        # accumulate arrows
        Xarr = []
        Varr = []
        Carr = []
        for k in range(len(traces)):
            trace = traces[k]
            if not start_included:
                trace = [start] + trace
            for t in range(1, len(trace)):
                i = trace[t-1]
                j = trace[t]
                Xarr.append(Xlo[i, :])
                Varr.append(Xlo[j, :] - Xlo[i, :])
                Carr.append(k)
        Xarr = np.stack(Xarr, 0)
        Varr = np.stack(Varr, 0)

        # draw all arrows
        plt.quiver(Xarr[:, 0], Xarr[:, 1], Varr[:, 0], Varr[:, 1], Carr, angles='xy', scale_units='xy', scale=1., width = 0.005)

    # plot means if given
    if means is not None:
        meanslo = np.dot(means - x, W.T) / scale
        clust_indices = np.unique(Y)
        plt.scatter(meanslo[:, 0], meanslo[:, 1], c = clust_indices, s = 100, marker = 'd')

        # decode into source code if model is given
        if model is not None:
            text_offset =  0.05 * (np.max(Xlo[:, 0]) - np.min(Xlo[:, 0]))
            # decode cluster means and plot them as well
            cluster_trees = decode_points(model, means, max_size = max_size)
            for k in range(means.shape[0]):
                nodes, adj = cluster_trees[k]
                var = variable_classifier.predict(nodes, adj) if variable_classifier is not None else None
                ast = python_ast_utils.tree_to_ast(nodes, adj, var)
                src = astor.to_source(ast)
                if src == '':
                    src = '<empty program>'
                plt.text(meanslo[k, 0] + text_offset, meanslo[k, 1], src, bbox=dict(facecolor='white', alpha=0.7), horizontalalignment = 'left', verticalalignment = 'bottom')
