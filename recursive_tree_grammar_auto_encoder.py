"""
Implements grammar auto-encoders as proposed in the paper.

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

import math
import numpy as np
import torch
import tree
import tree_grammar

class ParserRule(torch.nn.Module):
    """ A class to wrap all parameters of a ParserRule.

    More formally, the encoding h(A) for nonterminal A in the grammar rule
    A -> x(B_1, ..., B_m) is generated recursively via the following formula

    h(A) = tanh(W_1 * h(B_1) + ... + W_m * h(B_m) + b)

    where W_1, ..., W_m are weight matrices and b is a bias vector.

    If any of the symbols B_i is starred, W_i is applied recursively.

    Attributes
    ----------
    _nont: str
        The left-hand-side nonterminal A of this rule.
    _children: list
        The right-hand-side nonterminal symbols B_1, ..., B_m of the
        corresponding grammar rule.
    _rule_index: int
        The index of this rule in the original grammar.
    _dim: int
        The encoding dimensionality.
    _dim_hid: int (default = None)
        If given, each child is associated with a two-layer neural net with
        _dim_hid hidden neurons instead of a single-layer neural net.
    _Ws: class torch.nn.ParameterList
        One encoding matrix W per child.
    _b: class torch.nn.Paramter
        A bias vector.

    """
    def __init__(self, nont, children, rule_index, dim, dim_hid = None):
        super(ParserRule, self).__init__()
        self._children = children
        self._nont = nont
        self._rule_index = rule_index
        self._dim = dim
        self._dim_hid = dim_hid
        if self._dim_hid is None:
            w_in_dim = self._dim
        else:
            w_in_dim = self._dim_hid
            self._hidden = torch.nn.ModuleList()
            for c in range(len(self._children)):
                self._hidden.append(torch.nn.Linear(self._dim, self._dim_hid))
        self._Ws = torch.nn.ParameterList()
        for c in range(len(self._children)):
            self._Ws.append(torch.nn.Parameter(torch.zeros(self._dim, w_in_dim)))
        self._b  = torch.nn.Parameter(torch.zeros(self._dim))
        self.reset_parameters()

    def reset_parameters(self):
        """
        This is a weight initialization copied from torch.nn.modules.linear

        """
        max_fan_in = 1
        for c in range(len(self._children)):
            torch.nn.init.kaiming_uniform_(self._Ws[c], a=math.sqrt(5))
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self._Ws[c])
            if fan_in > max_fan_in:
                max_fan_in = fan_in
        bound = 1 / math.sqrt(max_fan_in)
        torch.nn.init.uniform_(self._b, -bound, bound)

class Encoder(torch.nn.Module):
    """ A tree parser for tree grammars. This parser
    performs a parsing process analogous to tree_grammar.TreeParser
    but in parallel performs a vectorial encoding of the input tree.

    This module can encode trees as vectors according to the rules of a
    given deterministic regular tree grammar. A grammar is called deterministic
    if no two production rules have the same right-hand-side.

    In more detail, we encode the production rules A -> x(B_1, ..., B_m)
    as a ModuleDict which maps symbols x to all ParserRules
    with x on the right-hand-side. We can then parse a tree bottom-up by
    replacing right-hand-sides of production rules with the nonterminal
    on the left until the entire tree is reduced to one nonterminal.
    In parallel with this process, we also create a vectorial representation
    for every nonterminal in the tree, such that the entire tree is
    represented by the vector for its starting symbol.

    Attributes
    ----------
    _grammar: class tree_grammar.TreeGrammar
        The original TreeGrammar.
    _rules: torch.nn.ModuleDict
        A dictionary mapping terminal symbols x to a list of ParserRules, one
        for each production rule of the form A -> x(B_1, ..., B_m).
    _dim: int
        The encoding dimensionality.
    _dim_hid: int (default = None)
        If given, each rule is associated with a two-layer neural net with
        _dim_hid hidden neurons instead of a single-layer neural net.
    _nonlin: class torch.nn.Module (default = torch.nn.Tanh())
        The nonlinearity to be applied after accumulating child encoding.

    """
    def __init__(self, grammar, dim, dim_hid = None, nonlin = torch.nn.Tanh()):
        super(Encoder, self).__init__()
        self._grammar = grammar
        self._dim = dim
        self._dim_hid = dim_hid
        self._nonlin = nonlin
        self._rules = torch.nn.ModuleDict()

        for left in grammar._rules:
            for r in range(len(grammar._rules[left])):
                sym, rights = grammar._rules[left][r]
                # construct a rule object
                sym_rule = ParserRule(left, rights, r, self._dim, self._dim_hid)
                # check if the right-hand-side nonterminal sequence is
                # deterministic
                tree_grammar.check_rule_determinism(rights)
                # check if a rule with the same right-hand-side symbol
                # already exists
                if sym not in self._rules:
                    self._rules[sym] = torch.nn.ModuleList()
                    self._rules[sym].append(sym_rule)
                else:
                    sym_rules = self._rules[sym]
                    # check if a rules with the same right hand side
                    # already exists
                    for sym_rule2 in sym_rules:
                        left2 = sym_rule2._nont
                        rights2 = sym_rule2._children
                        intersect = tree_grammar.rules_intersect(rights, rights2)
                        if intersect is not None:
                            right_strs = [str(right) for right in rights]
                            right_str = ', '.join(right_strs)
                            right_strs = [str(right) for right in rights2]
                            right_str2 = ', '.join(right_strs)
                            raise ValueError('The given grammar was ambiguous: There are two production rules with an intersecting right-hand side, namely %s -> %s(%s) and %s -> %s(%s), both accepting' % (left, sym, right_str, left2, sym, right_str2, str(intersect)))
                    # if that is not the case, append the new rule
                    sym_rules.append(sym_rule)

    def forward(self, nodes, adj):
        """ Synonym for self.parse """
        return self.parse(nodes, adj)

    def parse(self, nodes, adj):
        """ Parses the given input tree, i.e. computes the rule sequence
        which generates the given tree and a vectorial encoding for it.

        Parameters
        ----------
        nodes: list
            a node list for the input tree.
        adj: list
            an adjacency list for the input tree.

        Returns
        -------
        seq: list
            a rule sequence such that self._grammar.produce(seq) is equal to
            nodes, adj.
        h: class torch.Tensor
            A self._dim dimensional vectorial encoding of the input tree.

        Raises
        ------
        ValueError
            If the input is not a tree or not part of the language.

        """
        # retrieve the root
        r = tree.root(adj)
        # parse the input recursively
        nont, seq, h = self._parse(nodes, adj, r)
        # check if we got the starting symbol out
        if nont == self._grammar._start:
            # if so, return the sequence
            return seq, h
        else:
            # otherwise, return None
            raise ValueError('Parsing ended with symbol %s, which is not the starting symbol %s' % (str(nont), str(self._grammar._start)))

    def _parse(self, nodes, adj, i):
        # check if the current node is in the alphabet at all
        if nodes[i] not in self._grammar._alphabet:
            raise ValueError('%s is not part of the alphabet' % str(nodes[i]))
        # then, parse all the children
        actual_rights = []
        seqs = []
        hs   = []
        for j in adj[i]:
            # get the nonterminal and the rule sequence which
            # generates the jth subtree
            nont_j, seq_j, h_j = self._parse(nodes, adj, j)
            # append it to the child nont list
            actual_rights.append(nont_j)
            # append the rule sequence to the sequence which
            # generates the ith subtree
            seqs.append(seq_j)
            # and append the vectorial encoding
            hs.append(h_j)

        h = torch.zeros(self._dim)
        # retrieve the matching production rule for the current situation
        for rule in self._rules[nodes[i]]:
            left   = rule._nont
            r      = rule._rule_index
            rights = rule._children
            match  = tree_grammar.rule_matches(rights, actual_rights)
            if match is not None:
                if len(match) != len(rights):
                    raise ValueError('Internal error: Match length does not correspond to rule length')
                # build the rule sequence generating the current subtree.
                # we first use rule r
                seq = [r]
                # then, process the match entry by entry
                c = 0
                for a in range(len(rights)):
                    if isinstance(rights[a], str):
                        if rights[a].endswith('?'):
                            # if the ath nonterminal is optional, use a 1
                            # production rule if we matched something with
                            # this symbol and 0 otherwise.
                            if match[a]:
                                seq.append(1)
                                # then produce the current child
                                seq += seqs[c]
                                if self._dim_hid is not None:
                                    hs[c] = self._nonlin(rule._hidden[a](hs[c]))
                                h = torch.nn.functional.linear(hs[c], rule._Ws[a], h)
                                c += 1
                            else:
                                seq.append(0)
                            continue
                        if rights[a].endswith('*'):
                            # if the ath nonterminal is starred, use a 1
                            # production rule for every matched symbol
                            h_a = torch.zeros(self._dim)
                            for m in range(len(match[a])):
                                seq.append(1)
                                # then produce the matched child
                                seq += seqs[c]
                                h_a = h_a + hs[c]
                                c += 1
                            # finally, use a 0 rule to end the production
                            seq.append(0)
                            if self._dim_hid is not None:
                                h_a = self._nonlin(rule._hidden[a](h_a))
                            h = torch.nn.functional.linear(h_a, rule._Ws[a], h)
                            continue
                    # in all other cases, just append the production rules
                    # for the current child
                    seq += seqs[c]
                    if self._dim_hid is not None:
                        hs[c] = self._nonlin(rule._hidden[a](hs[c]))
                    h = torch.nn.functional.linear(hs[c], rule._Ws[a], h)
                    c += 1
                # complete the encoding by adding the bias and applying the nonlinearity
                return left, seq, self._nonlin(h + rule._b)
        # if no such rule exists, the parse fails
        raise ValueError('No rule with %s(%s) on the right-hand side exists' % (str(nodes[i]), str(actual_rights)))



class GrammarRule(torch.nn.Module):
    """ A class to wrap all parameters of a GrammarRule.

    More formally, the decoding for encoding h(A) of the grammar rule
    A -> x(B_1, ..., B_m) is generated recursively via the following formula

    h(B_j) = nonlin(W_j * g_j + b_j) where
    g_j    = g_{j-1} - h(B_{j-1}) and
    g_1    = h(A)

    where W_1, ..., W_m are weight matrices specific to this rule and where
    b_1, ..., b_m are bias vectors specific to this rule.

    The decision whether this rule is applied at all is made using a linear
    classifier on the encodings h(A), i.e. each rule has an 'affinity' vector
    v which is multiplied with h(A) and the rule with highest affinity is
    applied.

    Attributes
    ----------
    _children: list
        The right-hand-side nonterminal symbols B_1, ..., B_m of the
        corresponding grammar rule.
    _nont: str
        The left-hand-side nonterminal A of this rule.
    _dim: int
        The encoding dimensionality.
    _dim_hid: int (default = None)
        If given, each child is associated with a two-layer neural net with
        _dim_hid hidden neurons instead of a single-layer neural net.
    _child_layers: class torch.nn.ModuleList
        One linear layer per child.

    """
    def __init__(self, nont, children, dim, dim_hid = None, nonlin = torch.nn.Tanh()):
        super(GrammarRule, self).__init__()
        self._children = children
        self._nont = nont
        self._dim = dim
        self._dim_hid = dim_hid
        self._child_layers = torch.nn.ModuleList()
        for c in range(len(self._children)):
            if dim_hid is None:
                child_layer_c = torch.nn.Linear(self._dim, self._dim)
            else:
                child_layer_c = torch.nn.Sequential(torch.nn.Linear(self._dim, self._dim_hid), nonlin, torch.nn.Linear(self._dim_hid, self._dim))
            self._child_layers.append(child_layer_c)

class SimpleGRU(torch.nn.Module):
    """ A easier-to-use version of the gated recurrent unit (Cho et al., 2014).

    Attributes
    ----------
    _dim_in: int
        The input dimensionality.
    _dim_hid: int
        The state dimensionality.
    _gru: class torch.nn.GRU
        The actual GRU network.

    """
    def __init__(self, dim_in, dim_hid):
        super(SimpleGRU, self).__init__()
        self._dim_in = dim_in
        self._dim_hid = dim_hid
        self._gru = torch.nn.GRU(self._dim_in, self._dim_hid)

    def forward(self, x, h = None):
        """ Updates the state given the current state and the current input.

        Parameters
        ----------
        x: class torch.Tensor
            The current input with dimensionality self._dim_in
        h: class torch.Tensor (default = zero vector)
            The current state with dimensionality self._dim_hid.

        Returns
        -------
        h: class torch.Tensor
            The next state vwith dimensionalit self._dim_hid, computed as
            described above.

        """
        # fill state with zeros if not given
        if h is None:
            h = torch.zeros(self._dim_hid)
        # extend with empty dimensions
        h = h.unsqueeze(0).unsqueeze(1)
        # extend x with empty dimensions
        x = x.unsqueeze(0).unsqueeze(1)
        # compute the next state
        _, h = self._gru(x, h)
        # return it
        return h.squeeze(1).squeeze(0)


class Decoder(torch.nn.Module):
    """ A regular tree grammar, which is controlled by a neural network and
    can thus decode a vector back into a tree.

    Attributes
    ----------
    _grammar:  class tree_grammar.TreeGrammar
        A regular tree grammar.
    _dim: int
        The encoding dimensionality.
    _dim_hid: int (default = None)
        If given, each rule is associated with a two-layer neural net with
        _dim_hid hidden neurons instead of a single-layer neural net.
    _rules: class torch.nn.ModuleDict
        An analogue of self._grammar._rules, but filled with GrammarRule
        objects.
    _Vs: class torch.nn.ModuleDict
        A linear classifier for each nonterminal (including starred and
        optional ones) to decide which rule to use. This is trained based
        on data and not set immediately.
    _nonlin: class torch.nn.Module
        The nonlinearity to be applied after accumulating child encoding.

    """
    def __init__(self, grammar, dim, dim_hid = None, nonlin = torch.nn.Tanh()):
        super(Decoder, self).__init__()
        self._grammar = grammar
        self._dim = dim
        self._dim_hid = dim_hid
        # generate a GrammarRule object for each grammar rule
        self._rules = torch.nn.ModuleDict()
        for left in self._grammar._rules:
            self._rules[left] = torch.nn.ModuleList()
            for sym, rights in self._grammar._rules[left]:
                rec_net_rule = GrammarRule(left, rights, self._dim, self._dim_hid, nonlin)
                self._rules[left].append(rec_net_rule)
        # generate an additional GRU and a linear layer for each nonterminal to
        # handle starred rules
        self._nont_star_layers = torch.nn.ModuleDict()
        self._nont_star_grus = torch.nn.ModuleDict()
        for left in self._grammar._rules:
            for sym, rights in self._grammar._rules[left]:
                for right in rights:
                    if isinstance(right, str) and right.endswith('*'):
                        if right not in self._nont_star_grus:
                            self._nont_star_layers[right] = torch.nn.Linear(self._dim, self._dim)
                            self._nont_star_grus[right]   = SimpleGRU(self._dim, self._dim)
        # generate an additional linear layer for each nonterminal to
        # decide which rule to apply
        self._Vs = torch.nn.ModuleDict()
        for left in self._grammar._rules:
            self._Vs[left] = torch.nn.Linear(self._dim, len(self._grammar._rules[left]))
            # also add additional classification layers for starred and
            # optional nonterminals
            for sym, rights in self._grammar._rules[left]:
                for right in rights:
                    if isinstance(right, str) and (right.endswith('*') or right.endswith('?')):
                        if right not in self._Vs:
                            self._Vs[right] = torch.nn.Linear(self._dim, 2)
        # store nonlinearity
        self._nonlin = nonlin

    def decode(self, h, verbose = False, max_size = None, stochastic = False, start = None):
        """ Produces a tree according based on the rules of self._grammar
        and the given input encoding.

        Parameters
        ----------
        h: class torch.Tensor
            A self._dim dimensional encoding of the tree that should be decoded.
        verbose: bool (default = False)
            If set to true, this prints every rule decision and the predicted
            subtree encoding used for it.
        max_size: int (default = None)
            A maximum tree size to prevent infinite recursion.
        stochastic: bool (default = False)
            If set to true, the grammar rule in each step is not just the
            maximally activated rule but instead sampled from the softmax
            distribution over rules. This also makes decoding nondeterministic.
        start: nonterminal (default = self._grammar._start)
            The starting nonterminal symbol for the decoding.

        Returns
        -------
        nodes: list
            the node list of the produced tree.
        adj: list
              the adjacency list of the produced tree.
        seq: list
              the rule sequence generating the produced tree.

        """
        if len(h) != self._dim:
            raise ValueError('Expected a %d-dimensional input vector, but got %d dimensions!' % (self._dim, len(h)))
        if isinstance(max_size, int):
            max_size = [max_size]
        elif max_size is not None:
            raise ValueError('Expected either an integer or None as max_size, but got %s.' % str(max_size))

        if start is None:
            start = self._grammar._start

        # note that we create the tree in recursive format
        # We start with a placeholder tree to hold our starting symbol
        parent = tree.Tree('$', [tree.Tree(start)])
        seq = []
        # decode the tree recursively
        self._decode(parent, 0, h, seq, verbose, max_size, stochastic)
        # return the final tree in node list/adjacency list format
        if parent._children[0] is None:
            nodes = []
            adj   = []
        elif isinstance(parent._children[0], list):
            nodes = []
            adj   = []
            for subtree in parent._children[0]:
                nodes2, adj2 = subtree.to_list_format()
                nodes.append(nodes2)
                adj.append(adj2)
        else:
            nodes, adj = parent._children[0].to_list_format()
        return nodes, adj, seq

    def _decode(self, par, c, h, seq, verbose = False, max_size = None, stochastic = False):
        if max_size is not None :
            if max_size[0] <= 0:
               return
            max_size[0] -= 1

        # retrieve the current nonterminal
        A = par._children[c]._label

        # compute the affinity of all available rules to the current encoding
        # and take the rule which matches best
        scores = self._Vs[A](h)
        if not stochastic:
            r_max  = torch.argmax(scores)
        else:
            # if the sampling is stochastic, sample instead from the
            # softmax distribution
            scores = scores.detach().numpy()
            # subtract the max score for numerical stability
            max_score = np.max(scores)
            choice_probs = np.exp(scores - max_score)
            # normalize
            choice_probs /= np.sum(choice_probs)
            # choose randomly
            r_max  = np.random.choice(len(scores), size = 1, p = choice_probs)[0]
        seq.append(r_max)

        if isinstance(A, str):
            # if the nonterminal ends with a *, we have to handle a list
            if A.endswith('*'):
                lst_node = par._children[c]
                # here, two rules are possible:
                if r_max == 0:
                    # the zeroth rule completes the list and means
                    # that we replace the entire tree node with the
                    # child list
                    par._children[c] = lst_node._children
                    if verbose:
                        print('decided to close starred nonterminal %s based on code %s' % (A, str(h.tolist())))
                else:
                    # the first rule continues the list, which means
                    # that we append a new child and process that
                    if verbose:
                        print('decided to continue starred nonterminal %s based on code %s' % (A, str(h.tolist())))
                    lst_node._children.append(tree.Tree(A[:-1]))
                    # extract the encoding for the current child and decode it
                    h_c = self._nonlin(self._nont_star_layers[A](h))
                    self._decode(lst_node, len(lst_node._children)-1, h_c, seq, verbose, max_size, stochastic)
                    # update h and check the starred nonterminal again
                    h = self._nont_star_grus[A](h_c, h)
                    self._decode(par, c, h, seq, verbose, max_size, stochastic)
                return
            # if the nonterminal ands with a ?, we have to handle an optional
            # node
            elif A.endswith('?'):
                # here, two rules are possible
                if r_max == 0:
                    # the zeroth rule means we replace the nonterminal with
                    # None
                    par._children[c] = None
                    if verbose:
                        print('decided to omit optional nonterminal %s based on code %s' % (A, str(h.tolist())))
                else:
                    # the first rule means we replace the nonterminal
                    # with its non-optional version and process it then
                    if verbose:
                        print('decided to use optional nonterminal %s based on code %s' % (A, str(h.tolist())))
                    par._children[c]._label = A[:-1]
                    A = A[:-1]
                    self._decode(par, c, h, seq, verbose, max_size, stochastic)
                return
        # if the nonterminal is neither starred, nor optional, use the
        # regular processing
        rule = self._rules[A][r_max]
        sym, children = self._grammar._rules[A][r_max]
        if verbose:
            print('decided to apply rule %s -> %s(%s) based on code %s' % (A, sym, str(children), str(h.tolist())))
        # replace A in the input tree with sym(rights)
        subtree = tree.Tree(sym)
        par._children[c] = subtree
        # recursively process all children with a recurrent scheme
        for c in range(len(children)):
            subtree._children.append(tree.Tree(children[c]))
            # generate the predicted encoding for the current child
            h_c = self._nonlin(rule._child_layers[c](h))
            # recursively process the current child
            self._decode(subtree, c, h_c, seq, verbose, max_size, stochastic)
            # update h; TODO this may be smarter via a recurrent matrix
            h = h - h_c

    def produce(self, h, seq, start = None):
        """ Produces a tree based on the given rule sequence. For more
        information, refer to tree_grammar.TreeGrammar.produce.
        While producing the output tree, this function also produces a
        sequence of decodings, one for each rule application, which can
        be used for training.

        Parameters
        ----------
        h: class torch.Tensor
            A self._dim dimensional encoding of the tree that should be decoded.
        seq: list
              A ground truth sequence of rule indices.
        start: str (default = self._grammar._start)
            The nonterminal from which to start decoding.

        Returns
        -------
        nodes: list
            the node list of the produced tree.
        adj: list
            the adjacency list of the produced tree.
        nonts: list
            the sequence of nonterminals generated during production.
        hs: list
            A list of encodings based on which the ground truth rule indices
            should be chosen.

        Raises
        ------
        ValueError
            If the given sequence refers to a terminal entry at some point, if
            the rule does not exist, or if there are still nonterminal symbols
            left after production.

        """
        if start is None:
            start = self._grammar._start
        elif start not in self._grammar._nonts:
            raise ValueError('%s is not a valid nonterminal for this grammar.' % str(start))
        # note that we create the tree in recursive format
        # We start with a placeholder tree for our starting symbol
        parent = tree.Tree('$', [tree.Tree(start)])
        # put this on the stack with the child index we consider
        stk = [(parent, 0, h)]
        # initialize decoding sequence
        hs  = []
        # and the nonterminal sequence
        nonts = []
        # iterate over all rule indices
        for r in seq:
            # throw an error if the stack is empty
            if not stk:
                raise ValueError('There is no nonterminal left but the input rules sequence has not ended yet')
            # pop the current parent, child index, and encoding
            par, c, h = stk.pop()
            hs.append(h)
            # retrieve the current nonterminal
            A = par._children[c]._label
            nonts.append(A)

            # if the nonterminal ends with a *, we have to handle a list
            if isinstance(A, str) and A.endswith('*'):
                lst_node = par._children[c]
                # here, two rules are possible:
                if r == 0:
                    # the zeroth rule completes the list and means
                    # that we replace the entire tree node with the
                    # child list
                    par._children[c] = par._children[c]._children
                    continue
                elif r == 1:
                    # the first rule continues the list, so we first extract
                    # the encoding for the new child
                    h_c = self._nonlin(self._nont_star_layers[A](h))
                    # and we update h
                    h = self._nont_star_grus[A](h_c, h)
                    # then, we put the current nonterminal on the stack
                    # again
                    stk.append((par, c, h))
                    # and we create another nonterminal child and put
                    # it on the stack
                    lst_node._children.append(tree.Tree(A[:-1]))
                    stk.append((lst_node, len(lst_node._children) - 1, h_c))
                    continue
                else:
                    raise ValueError('List nodes only accept rules 0 (stop) and 1 (continue)')
            # if the nonterminal ands with a ?, we have to handle an optional
            # node
            if isinstance(A, str) and A.endswith('?'):
                # here, two rules are possible
                if r == 0:
                    # the zeroth rule means we replace the nonterminal with
                    # None
                    par._children[c] = None
                    continue
                elif r == 1:
                    # the first rule means we replace the nonterminal
                    # with its non-optional version and push it on the
                    # stack again
                    par._children[c]._label = A[:-1]
                    stk.append((par, c, h))
                    continue
                else:
                    raise ValueError('Optional nodes only accept rules 0 (stop) and 1 (continue)')

            # get the current rule
            sym, rights = self._grammar._rules[A][r]
            rule        = self._rules[A][r]
            # replace A in the input tree with sym(rights)
            subtree = par._children[c]
            subtree._label = sym
            # and produce all child encodings
            child_encodings = []
            for c in range(len(rights)):
                subtree._children.append(tree.Tree(rights[c]))
                # generate the predicted encoding for the current child
                h_c = self._nonlin(rule._child_layers[c](h))
                child_encodings.append(h_c)
                # update h; TODO this may be smarter via a recurrent matrix
                h = h - h_c
            # push all new child nonterminals onto the stack
            for c in range(len(rights)-1, -1, -1):
                stk.append((subtree, c, child_encodings[c]))
        # throw an error if the stack is not empty yet
        if stk:
            raise ValueError('There is no rule left anymore but there are still nonterminals left in the tree')
        # return the final tree in node list adjacency list format
        nodes, adj = parent._children[0].to_list_format()
        return nodes, adj, nonts, hs

    def compute_loss(self, h, seq):
        """ Computes the crossentropy classification loss of this Decoder
        to predict the given sequence of rules from the given encoding vector.

        Parameters
        ----------
        h: class torch.Tensor
            A self._dim dimensional encoding of the tree that should be
            decoded.
        seq: list
            A ground truth sequence of rule indices.

        Returns
        -------
        loss: class torch.Tensor
            A scalar torch.Tensor containing the crossentropy classification
            loss between the ground truth rules and the predicted rules of
            this network.

        Raises
        ------
        ValueError
            If the given sequence refers to a terminal entry at ome point, if
            the rule does not exist, or if there are still nonterminal symbols
            left after production.

        """
        # in a first step, produce the target tree with the given rule sequence
        _, _, nonts, hs = self.produce(h, seq)
        # compute the predicted sequence of rules
        pred = []
        for t in range(len(nonts)):
            # compute the affinity scores for the current nonterminal and the
            # computed nonterminal representation
            scores = self._Vs[nonts[t]](hs[t])
            # append them to the prediction sequence
            pred.append(scores)

        # pad the predictions with zero to be length-consistent
        pred = torch.nn.utils.rnn.pad_sequence(pred, True)
        # convert ground truth to tensor
        seq  = torch.tensor(seq, dtype=torch.long)

        # compute loss
        return torch.nn.functional.cross_entropy(pred, seq)


class TreeGrammarAutoEncoder(torch.nn.Module):
    """ A tree grammar variational autoencoder consisting of a NeuralParser and
    a Decoder as well as in between layers which implement the
    variational auto-encoder

    Attributes
    ----------
    _grammar:  class tree_grammar.TreeGrammar
        A regular tree grammar.
    _dim: int
        The encoding dimensionality for the tree networks.
    _dim_vae: int (default = self._dim)
        The encoding dimensionality of the variational auto-encoder.
    _dim_hid: int (default = None)
        The hidden dimensionality for child-to-parent and parent-to-child
        neural nets. If None, single-layer nets are used.
    _enc: class Encoder
        The Encoder.
    _dec: class Decoder
        The Decoder.
    _vae_enc: class torch.nn.Linear
        A _dim to (2 * _dim_vae) linear layer to map from a tree code to the
        mean and standard deviation of the VAE code.
    _vae_dec: class torch.nn.Linear
        A _dim_vae to _dim linear layer to map from a VAE code to a tree code.
    _nonlin: class torch.nn.Module
        The nonlinearity.

    """
    def __init__(self, grammar, dim, dim_vae = None, dim_hid = None, nonlin = torch.nn.Tanh()):
        super(TreeGrammarAutoEncoder, self).__init__()
        self._grammar = grammar
        self._enc     = Encoder(grammar, dim, dim_hid, nonlin)
        self._dec     = Decoder(grammar, dim, dim_hid, nonlin)
        self._dim     = dim
        if dim_vae is None:
            self._dim_vae = self._dim
        else:
            self._dim_vae = dim_vae
        self._vae_enc = torch.nn.Linear(self._dim, 2 * self._dim_vae)
        self._vae_dec = torch.nn.Linear(self._dim_vae, self._dim)
        self._nonlin = nonlin

    def encode(self, nodes, adj):
        """ Encodes the given tree as a vector and parses it at the same time.

        Parameters
        ----------
        nodes: list
            a node list for the input tree.
        adj: list
            an adjacency list for the input tree.

        Returns
        -------
        seq: list
            a rule sequence such that self._grammar.produce(seq) is equal to
            nodes, adj.
        h: class torch.Tensor
            A self._dim_vae dimensional vectorial encoding of the input tree.
            This corresponds to the mean of the variational auto-encoder.

        Raises
        ------
        ValueError
            If the input is not a tree or not part of the language.

        """
        # parse and code the tree
        seq, h = self._enc.parse(nodes, adj)
        # use the VAE encoder
        mu_and_sigma = self._vae_enc(h)
        # return only the mean
        mu = mu_and_sigma[:self._dim_vae]
        return seq, mu

    def decode(self, z, verbose = False, max_size = None, stochastic = False, start = None):
        """ Decodes the given vector back into a tree. This is only
        available if fit has been called.

        Parameters
        ----------
        z: class torch.Tensor
            A self._dim_vae dimensional encoding of the tree that should be
            decoded.
        verbose: bool (default = False)
            If set to true, this prints every rule decision and the predicted
            subtree encoding used for it.
        max_size: int (default = None)
            A maximum tree size to prevent infinite recursion.
        stochastic: bool (default = False)
            If set to true, the grammar rule in each step is not just the
            maximally activated rule but instead sampled from the softmax
            distribution over rules. This also makes decoding nondeterministic.
        start: nonterminal (default = self._grammar._start)
            The starting nonterminal symbol for the decoding.

        Returns
        -------
        nodes: list
            the node list of the produced tree.
        adj: list
            the adjacency list of the produced tree.
        seq: list
            the rule sequence generating the produced tree.

        """
        # use the VAE decoder
        h = self._nonlin(self._vae_dec(z))
        # use the tree decoder
        return self._dec.decode(h, verbose, max_size, stochastic = stochastic, start = start)

    def compute_loss(self, nodes, adj, beta = 1., sigma_scaling = 1.):
        """ Computes the variational auto-encoding loss for the given tree.

        Parameters
        ----------
        nodes: list
            the node list of the input tree.
        adj: list
            the adjacency list of the input tree.
        beta: float (default = 1.)
            the regularization constant for the VAE.
        sigma_scaling: float (default = 1.)
            A scaling factor to reduce the noise introduced by the VAE.

        Returns
        -------
        loss: class torch.Tensor
            a torch scalar containing the VAE loss, i.e. reconstruction-loss
            + beta * (mu^2 + sigma^2 - log(sigma^2) - 1)

        Raises
        ------
        ValueError
            If the input is not a tree or not part of the language.

        """
        # parse and code the tree
        seq, h = self._enc.parse(nodes, adj)
        # use the VAE encoder
        mu_and_logvar = self._vae_enc(h)
        # extract mean and standard deviations
        mu = mu_and_logvar[:self._dim_vae]
        logvar = mu_and_logvar[self._dim_vae:]
        sigmas = torch.exp(logvar * 0.5)
        # construct a random code
        if sigma_scaling > 0.:
            z = torch.randn(self._dim_vae) * sigmas * sigma_scaling + mu
        else:
            z = mu
        # use the VAE decoder
        h = self._nonlin(self._vae_dec(z))
        # compute the reconstruction loss
        loss_obj = self._dec.compute_loss(h, seq)
        # add the regularization
        if beta > 0.:
            mu2 = torch.pow(mu, 2)
            sigmas2 = torch.exp(logvar)
            loss_obj = loss_obj + beta * torch.sum(mu2 + sigmas2 - logvar - 1)
        return loss_obj
