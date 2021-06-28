"""
Implements a simplified version of the recursive tree grammar autoencoder,
which uses a shared GRU for encoding and a shared GRU for decoding
across all symbols.

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
__copyright__ = 'Copyright 2021, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '0.2.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'benjamin.paassen@sydney.edu.au'

import torch
import tree
import tree_grammar

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
            The next state with dimensionality self._dim_hid, computed as
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

class RTGAE(torch.nn.Module):
    """ An autoencoder for trees over a given grammar.

    Parameters
    ----------
    grammar: class tree_grammar.TreeGrammar
        The grammar from which trees are generated
    dim: int
        The encoding dimensionality.

    Attributes
    ----------

    """
    def __init__(self, grammar, dim):
        super(RTGAE, self).__init__()
        self.grammar = grammar
        self.dim = dim

        # set up a tree parser for the given grammar. We need this for
        # training the autoencoder
        self.parser_ = tree_grammar.TreeParser(grammar)

        # set up a map from symbols to indices
        self.symbol_to_idx_ = {}
        symbols = list(self.grammar._alphabet.keys())
        for i in range(len(symbols)):
            self.symbol_to_idx_[symbols[i]] = i

        # set up a GRU to encode siblings
        self.enc_siblings_ = torch.nn.GRU(self.dim, self.dim)
        # set up a GRU to encode to parents
        self.enc_parents_  = SimpleGRU(len(symbols), self.dim)
        # set up a GRU to decode siblings
        self.dec_siblings_ = SimpleGRU(self.dim, self.dim)
        self.dec_siblings_out_ = torch.nn.Linear(self.dim, self.dim)
        # set up a GRU to decode from parents
        self.dec_parents_  = SimpleGRU(len(symbols), self.dim)
        # set up linear layers to decide which grammar rule to take
        # from a given nonterminal symbol
        self.dec_cls_ = torch.nn.ModuleDict()
        for nont in self.grammar._nonts:
            self.dec_cls_[nont] = torch.nn.Linear(self.dim, len(self.grammar._rules[nont]))
        # also set up classifiers for optional and starred rules
        for lhs in self.grammar._rules:
            for rhs in self.grammar._rules[lhs]:
                for nont in rhs[1]:
                    if nont not in self.dec_cls_ and isinstance(nont, str) and (nont.endswith('*') or nont.endswith('?')):
                        self.dec_cls_[nont] = torch.nn.Linear(self.dim, 2)


    def encode(self, x):
        """ Encodes the given input tree recursively by first
        encoding the child sequence to a vector and then the state
        representing the child sequence to a parent state.

        Parameters
        ----------
        x: class tree.Tree
            An input tree.

        Returns
        -------
        h: class torch.Tensor
            A self.dim dimensional vectorial encoding of the input tree.

        """
        # first encode the sequence of children recursively
        if len(x._children) == 0:
            h_child = torch.zeros(self.dim)
        else:
            H_child = torch.zeros(len(x._children), self.dim)
            for k in range(len(x._children)):
                H_child[k, :] = self.encode(x._children[k])
            _, h_child = self.enc_siblings_(H_child.unsqueeze(1))
            h_child = h_child.squeeze(1).squeeze(0)
        # then, encode to a parent representation
        label_encoding = torch.zeros(len(self.symbol_to_idx_))
        label_encoding[self.symbol_to_idx_[x._label]] = 1.
        h_parent = self.enc_parents_(label_encoding, h_child)
        return h_parent


    def decode(self, h, max_size = None):
        """ Decodes the given vector into a tree.

        Parameters
        ----------
        h: class torch.Tensor
            A self.dim dimensional vectorial encoding of some tree.
        max_size: int (default = None)
            A maximum number of decoding steps to prevent endless recursion.

        Returns
        -------
        x: class tree.Tree
            The decoded tree.

        """
        x, N = self.decode_(h, self.grammar._start, max_size)
        return x

    def decode_(self, h, nont, max_size = None):
        """ Decodes the given vector into a tree.

        Parameters
        ----------
        h: class torch.Tensor
            A self.dim dimensional vectorial encoding of some tree.
        nont: str
            The nonterminal symbol from which decoding shall start.
        max_size: int (default = None)
            A maximum number of decoding steps to prevent endless recursion.

        Returns
        -------
        x: class tree.Tree
            The decoded tree.
        N: int
            The size of the decoded tree.

        """
        # first, decide the grammar rule that shall be applied
        r = torch.argmax(self.dec_cls_[nont](h))
        rule = self.grammar._rules[nont][r]
        # generate a new tree object with the label determined by the
        # grammar rule
        label = rule[0]
        child_nonts = rule[1]
        x = tree.Tree(label)
        # map from parent to child state
        label_encoding = torch.zeros(len(self.symbol_to_idx_))
        label_encoding[self.symbol_to_idx_[x._label]] = 1.
        h_child = self.dec_parents_(label_encoding, h)
        # generate all children that are determined by the grammar rule
        child_nonts = rule[1]
        N = 1
        if max_size is not None:
            max_size -= 1
        for k in range(len(child_nonts)):
            if isinstance(child_nonts[k], str) and child_nonts[k].endswith('*'):
                # if the next child is starred, continue decoding children for
                # this nonterminal until the classifier says otherwise
                while max_size is None or max_size > 0:
                    continue_flag = torch.argmax(self.dec_cls_[child_nonts[k]](h_child))
                    if continue_flag == 0:
                        break
                    # infer state for the next child
                    hk = self.dec_siblings_out_(h_child)
                    # decode it recursively and append it
                    y, Nk = self.decode_(hk, child_nonts[k][:-1], max_size)
                    x._children.append(y)
                    # adjust the remaining maximum size
                    if max_size is not None:
                        max_size -= Nk
                    N += Nk
                    # update the state
                    h_child = self.dec_siblings_(hk, h_child)
                # once all children for the starred nonterminal are decoded,
                # continue with the next child nonterminal
                continue
            elif isinstance(child_nonts[k], str) and child_nonts[k].endswith('?'):
                # if the next child is optional, check if it shall be decoded
                continue_flag = torch.argmax(self.dec_cls_[child_nonts[k]](h_child))
                if continue_flag == 0:
                    # if not, continue right away
                    continue
                nont = child_nonts[k][:-1]
            else:
                nont = child_nonts[k]
            # if the max size is exceeded, leave the nonterminals
            if max_size is not None and max_size <= 0:
                x._children.append(tree.Tree(nont))
                continue
            # at this point, we know that the child shall be decoded regularly.
            # First, we infer the state for the next child
            hk = self.dec_siblings_out_(h_child)
            # then, we decode it recursively and append it
            y, Nk = self.decode_(hk, nont, max_size)
            x._children.append(y)
            # adjust the remaining maximum size
            if max_size is not None:
                max_size -= Nk
            N += Nk
            # update the state
            h_child = self.dec_siblings_(hk, h_child)
        # return
        return x, N

    def decode_forced_(self, h, nont, seq, t, score_list, decoding_list = None):
        """ Decodes the given vector into a tree, forced by the given sequence
        of rules and accumulates the rule scores of the model along the way.

        Parameters
        ----------
        h: class torch.Tensor
            A self.dim dimensional vectorial encoding of some tree.
        nont: str
            The nonterminal symbol from which decoding shall start.
        seq: list
            A sequence of grammar rules.
        t: int
            The current index in seq.
        score_list: list
            The output list of scores.
        decoding_list: list (default = None)
            The output list for decodings along the way.

        Returns
        -------
        t: int
            The index in seq after decoding h.

        """
        # first, decide the current scores
        if decoding_list is not None:
            decoding_list.append(h)
        score_list.append(self.dec_cls_[nont](h))
        # then, retrieve the rule that ought to be applied
        r = seq[t]
        t += 1
        rule = self.grammar._rules[nont][r]
        # generate a new tree object with the label determined by the
        # grammar rule
        label = rule[0]
        child_nonts = rule[1]
        # map from parent to child state
        label_encoding = torch.zeros(len(self.symbol_to_idx_))
        label_encoding[self.symbol_to_idx_[label]] = 1.
        h_child = self.dec_parents_(label_encoding, h)
        # generate all children that are determined by the grammar rule
        child_nonts = rule[1]
        for k in range(len(child_nonts)):
            if isinstance(child_nonts[k], str) and child_nonts[k].endswith('*'):
                # if the next child is starred, continue decoding children for
                # this nonterminal until the rule sequence says otherwise
                score_list.append(self.dec_cls_[child_nonts[k]](h_child))
                while seq[t] == 1:
                    t += 1
                    # infer state for the next child
                    hk = self.dec_siblings_out_(h_child)
                    # decode the current child
                    t = self.decode_forced_(hk, child_nonts[k][:-1], seq, t, score_list, decoding_list)
                    # update the state
                    h_child = self.dec_siblings_(hk, h_child)
                    score_list.append(self.dec_cls_[child_nonts[k]](h_child))
                t += 1
                # once all children for the starred nonterminal are decoded,
                # continue with the next child nonterminal
                continue
            elif isinstance(child_nonts[k], str) and child_nonts[k].endswith('?'):
                # if the next child is optional, check if it shall be decoded
                score_list.append(self.dec_cls_[child_nonts[k]](h_child))
                if seq[t] == 0:
                    t += 1
                    continue
                t += 1
                nont = child_nonts[k][:-1]
            else:
                nont = child_nonts[k]
            # at this point, we know that the child shall be decoded regularly.
            # First, we infer the state for the next child
            hk = self.dec_siblings_out_(h_child)
            # then, we decode it recursively
            t = self.decode_forced_(hk, nont, seq, t, score_list, decoding_list)
            # update the state
            h_child = self.dec_siblings_(hk, h_child)
        # return
        return t

    def compute_loss(self, x):
        """ Autoencodes the given tree and computes the classification loss
        between selecting the correct node labels and the choice of the current
        model.

        Parameters
        ----------
        x: class tree.Tree
            An input tree.

        Returns
        -------
        loss: class torch.Tensor
            The classification loss for choosing the correct node labels.

        """
        # encode the tree
        h = self.encode(x)
        # find the actual rule sequence that should be chosen by using a
        # tree parser
        nont, seq = self.parser_.parse_tree(x)
        # find the rules that the current autoencoder would have chosen instead
        scores = []
        self.decode_forced_(h, nont, seq, 0, scores)
        # transform the actual score list and the desired chosen rules
        # into torch tensors
        scores = torch.nn.utils.rnn.pad_sequence(scores, True)
        seq    = torch.tensor(seq, dtype=torch.long)
        # compute crossentropy loss
        loss = torch.nn.functional.cross_entropy(scores, seq, reduction = 'sum')
        # return it
        return loss
