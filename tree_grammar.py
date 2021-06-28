"""
Implements a tree grammar.

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
__version__ = '0.2.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'benjamin.paassen@sydney.edu.au'

import copy
import tree

class TreeGrammar:
    """ A regular tree grammar, defined by an arity alphabet of terminal
    symbols, a set of nonterminal symbols, a starting nonterminal symbol
    and a set of production rules of the form

    A -> x(B_1, ..., B_m)

    where A, B_1, ..., B_m are nonterminal symbols, x is a terminal symbol
    and m is the arity of x.

    Attributes
    ----------
    _alphabet: dict
        A dictionary mapping terminal symbols to arities. If a list is given
        as input, it is transformed into a dict.
    _nonts: list
        A set or list of nonterminal symbols.
    _start: str
        An element of the _nonts set.
    _rules: dict
        A dictionary mapping nonterminal symbols A to tuples
        (x, [B_1, ..., B_m]) for each production rule of the form
        A -> x(B_1, ..., B_m).

    """
    def __init__(self, alphabet, nonts, start, rules):
        # check if the alphabet is a dictionary mapping to integers
        if isinstance(alphabet, dict):
            for sym in alphabet:
                if(not isinstance(alphabet[sym], int)):
                    raise ValueError('The alphabet entry for symbol %s is not an integer!' % sym)
            self._alphabet = alphabet
        else:
            # otherwise, we infer the symbol-to-child number mapping
            # from the rule set
            self._alphabet = {}
            for left in rules:
                for (sym, rights) in rules[left]:
                    self._alphabet[sym] = len(rights)
            for sym in self._alphabet:
                if sym not in alphabet:
                    raise ValueError('%s was the terminal symbol of the right-hand-side of a production rule but is not in the alphabet %s' % (str(sym), str(alphabet)))

        # check if the starting nonterminal is part of the nonterminal set
        nonts = frozenset(nonts)
        if(start not in nonts):
            raise ValueError('The starting nonterminal symbol %s is not part of the nonterminal set %s' % (str(start), str(nonts)))
        self._nonts = nonts
        self._start = start
        # check the rules, which should be tuples of terminals and nonterminal
        # lists
        for left in rules:
            if(left not in nonts):
                raise ValueError('%s was the left hand side of a production rule but is not in the nonterminal set %s' % (str(left), str(nonts)))
            for (sym, rights) in rules[left]:
                if(sym not in self._alphabet):
                    raise ValueError('%s was the terminal symbol of the right-hand-side of a production rule but is not in the alphabet %s' % (str(sym), str(self._alphabet)))
                for right in rights:
                    if(isinstance(right, str) and (right.endswith('*') or right.endswith('?'))):
                        right = right[:-1]
                    if(right not in nonts):
                        raise ValueError('%s was on the left hand side of a production rule but is not in the nonterminal set %s' % (str(right), str(nonts)))
                if(len(rights) != self._alphabet[sym]):
                    right_strs = [str(right) for right in rights]
                    raise ValueError('The production rule %s -> %s(%s) produces %d nonterminal symbols, but %s requires %d children' % (str(left), str(sym), ', '.join(right_strs), len(rights), str(sym), self._alphabet[sym]))
        self._rules = rules

    def produce(self, seq, start = None, max_size = None, label_nonterminals_and_rules = False):
        """ Creates a tree according to the given sequence of rules. The given
        input sequence is actually a sequence of rule indices for the current
        nonterminal symbol. In particular, the production works as follows.
        We start from the trivial tree consisting only of the starting nonterminal
        and we push the starting nonterminal on a stack.

        Then, until the stack is empty, we pop the next nonterminal A from the
        stack, get the current rule index r from seq, retrieve the production
        rule A -> x(B_1, ..., B_m) as self._rules[A][r], replace A in the tree
        with x(B_1, ..., B_m), and push B_m, ..., B_1 onto the stack.


        Parameters
        ----------
        seq: list
            A sequence of rule indices as specified above.
        start: str (default = self._start)
            The starting nonterminal.
        max_size: int (default = None)
            A maximum number of rules that get used.
        label_nonterminals_and_rules: bool (default = False)
            If set to true, the output tree is not labeled with terminal
            symbols but instead with nonterminal symbols and the rule that is
            applied.

        Returns
        -------
        tree: class tree.Tree
            The output tree.

        Raises
        ------
        ValueError
            If the given sequence refers to a terminal entry at some point, if
            the rule does not exist, or if there are still nonterminal symbols
            left after production.

        """
        if(start is None):
            start = self._start
        elif(start not in self._nonts):
            raise ValueError('%s is not a valid nonterminal for this grammar.' % str(start))
        if max_size is not None and len(seq) > max_size:
            max_size_applies = True
            seq = seq[:max_size]
        else:
            max_size_applies = False

        # note that we create the tree in recursive format
        # We start with a placeholder tree for our starting symbol
        parent = tree.Tree('$', [tree.Tree(start)])
        # put this on the stack with the child index we consider
        stk = [(parent, 0)]
        # iterate over all rule indices
        for r in seq:
            # throw an error if the stack is empty
            if(not stk):
                raise ValueError('There is no nonterminal left but the input rules sequence has not ended yet')
            # pop the current parent and child index
            par, c = stk.pop()
            # retrieve the current nonterminal
            A = par._children[c]._label
            # if the nonterminal ends with a *, we have to handle a list
            if(isinstance(A, str) and A.endswith('*')):
                lst_node = par._children[c]
                # here, two rules are possible:
                if(r == 0):
                    # the zeroth rule completes the list and means
                    # that we replace the entire tree node with the
                    # child list
                    par._children[c] = par._children[c]._children
                    continue
                elif(r == 1):
                    # the first rule continues the list, which means
                    # that we put the current nonterminal on the stack
                    # again
                    stk.append((par, c))
                    # and we create another nonterminal child and put
                    # it on the stack
                    lst_node._children.append(tree.Tree(A[:-1]))
                    stk.append((lst_node, len(lst_node._children) - 1))
                    continue
                else:
                    raise ValueError('List nodes only accept rules 0 (stop) and 1 (continue)')
            # if the nonterminal ands with a ?, we have to handle an optional
            # node
            if(isinstance(A, str) and A.endswith('?')):
                # here, two rules are possible
                if(r == 0):
                    # the zeroth rule means we replace the nonterminal with
                    # None
                    par._children[c] = None
                    continue
                elif(r == 1):
                    # the first rule means we replace the nonterminal
                    # with its non-optional version and push it on the
                    # stack again
                    par._children[c]._label = A[:-1]
                    stk.append((par, c))
                    continue
                else:
                    raise ValueError('Optional nodes only accept rules 0 (stop) and 1 (continue)')

            # get the current rule
            sym, rights = self._rules[A][r]
            # replace A in the input tree with sym(rights)
            subtree = par._children[c]
            if not label_nonterminals_and_rules:
                subtree._label = sym
            else:
                subtree._label = (A, r)
            # push all new child nonterminals onto the stack and
            # to the tree
            for c in range(len(rights)-1, -1, -1):
                subtree._children.insert(0, tree.Tree(rights[c]))
                stk.append((subtree, c))
        # throw an error if the stack is not empty yet
        if not max_size_applies and stk:
            raise ValueError('There is no rule left anymore but there are still nonterminals left in the tree')
        # return the final tree
        return parent._children[0]

    def __str__(self):
        out = 'Alphabet: %s\nNonterminals: %s\nStarting Symbol: %s\nRules: {' % (str(self._alphabet), str(self._nonts), str(self._start))
        rule_strs = []
        for left in self._rules:
            for (sym, rights) in self._rules[left]:
                right_strs = [str(right) for right in rights]
                rule_strs.append('%s -> %s(%s)' % (str(left), str(sym), ', '.join(right_strs)))
        return out + ',\n'.join(rule_strs) + '}'

    def __repr__(self):
        return self.__str__()

def check_rule_determinism(As):
    """ Checks whether the given nonterminal sequence can be
    used directly as a deterministic finite-state automaton,
    considering * and ? concepts.

    Parameters
    ----------
    As: list
        a nonterminal sequence.

    Raises
    ------
    ValueError
        if the sequence is not deterministic.

    """
    # maintain a set of symbols which could introduce ambiguity
    # in the current step
    possible_ambiguities = set()
    for A in As:
        # if A does not end with a * or ? symbol, it cuts off all previous
        # possibilities for ambiguity
        if(not isinstance(A, str) or not (A.endswith('?') or A.endswith('*')) ):
            possible_ambiguities = set()
            continue
        # otherwise, it may introduce ambiguity and we add it to the set
        A = A[:-1]
        if(A in possible_ambiguities):
            raise ValueError('The given sequence %s is ambiguous because %s may belong to multiple slots.' % (str(As), A))
        possible_ambiguities.add(A)

def rule_matches(As, Bs):
    """ Checks whether the second nonterminal sequence matches the
    specification of the first, considering the * and ? concepts.

    As: list
        A nonterminal sequence, possibly including * and ? concepts.
    Bs: list
        A nonterminal sequence without these concepts.

    Returns
    -------
    Bs_new: list
        A new version of Bs of the same length as As which assigns to each
        element in As the corresponding elements in Bs or None if the two
        sequences don't match.

    """
    # if both sequences are strictly equal, we can end the test right
    # away
    if(As == Bs):
        return Bs
    # otherwise, if A is empty, this fails
    if(not As):
        return None
    # translate As symbol by symbol to a finite-state automaton and
    # use that to parse Bs
    a = 0
    a_tainted = True
    b = 0
    Bs_new = []
    while(a < len(As) and b < len(Bs)):
        # check the current state of the finite-state automaton
        if(a_tainted):
            A = As[a]
            is_optional = False
            is_star = False
            if(isinstance(A, str)):
                if(A.endswith('?')):
                    is_optional = True
                    A = A[:-1]
                elif(A.endswith('*')):
                    is_star = True
                    A = A[:-1]
            a_tainted = False
        # get the current nonterminal in Bs
        B = Bs[b]
        if(B != A):
            # if B does _not_ match with A, check if A is starred or optional
            if(is_optional or is_star):
                # If so, we can just try the next symbol in As
                Bs_new.append([])
                a += 1
                a_tainted = True
            else:
                # otherwise, there is no match
                return None
        else:
            # if B _does_ match with A, check if A is starred or optional
            if(is_star or is_optional):
                # if A is starred, check the special case of constructs like
                # A*AA ... where we need to look ahead and ensure that enough
                # As are left
                # look how many more As we have in A
                a2 = a + 1
                while(a2 < len(As) and As[a2] == A):
                    a2 += 1
                # look how many more As we have in B
                b2 = b + 1
                while(b2 < len(Bs) and Bs[b2] == A):
                    b2 += 1
                # if there are too few As in Bs, there is no match
                num_A = a2 - a - 1
                num_B = b2 - b
                if(num_B < num_A):
                    return None
                # create a list of all symbols that the current symbol
                # needs to parse
                need_to_parse = []
                for c in range(num_B - num_A):
                    need_to_parse.append(Bs[b + c])
                # if the current A symbol is only optional and
                # need_to_parse is longer than 1, there is no match
                if(is_optional and len(need_to_parse) > 1):
                    return None
                Bs_new.append(need_to_parse)
                a += 1
                b += num_B - num_A
                # and then process all subsequent matches right away
                for c in range(num_A):
                    Bs_new.append(Bs[b + c])
                a += num_A
                b += num_A
            else:
                # otherwise, just add B to the new version
                Bs_new.append(B)
                a += 1
                b += 1
            a_tainted = True
    # if we did not process all symbols in Bs yet, there is no match
    if(b < len(Bs)):
        return None
    # after we processed all Bs, check of only optional symbols in A
    # remain
    while(a < len(As)):
        A = As[a]
        if(not isinstance(A, str) or not (A.endswith('?') or A.endswith('*')) ):
            return None
        a += 1
        Bs_new.append([])
    # if the process never failed, we return True
    return Bs_new

def reduce_rule(As):
    """ Removes all starred and optional nonterminals from a nonterminal
    sequence.

    Parameters
    ----------
    As: list
        A nonterminal sequence, possibly including * and ? concepts.

    Returns
    -------
    out: list
        A copy of that sequence without these concepts.

    """
    out = []
    for A in As:
        if(isinstance(A, str) and (A.endswith('*') or A.endswith('?'))):
            continue
        out.append(A)
    return out

def rules_intersect(As, Bs):
    """ Checks if the two nonterminal sequences, both of which may contain
    * or ? symbols, intersect, in the sense that there exists a sequence
    of nonterminals Cs, such that rule_matches(As, Cs) AND rule_matches(Bs, Cs).

    Parameters
    ----------
    As: list
        A nonterminal sequence, possibly including * and ? concepts.
    Bs: list
        A nonterminal sequence, possibly including * and ? concepts.

    Return
    ------
    red: list
        The shortest sequence Cs such that rule_matches(As, Cs) AND
        rule_matches(Bs, Cs) or None if no such sequence exists.

    """
    # reduce both input sequence and check whether they match the
    # respective other sequence
    As_red = reduce_rule(As)
    Bs_red = reduce_rule(Bs)
    # if both are equal, return right away
    if(As_red == Bs_red):
        return As_red
    # make sure we test the shorter one first
    if(len(Bs_red) < len(As_red)):
        As_red, Bs_red = Bs_red, As_red
        As, Bs = Bs, As
    # then check matches
    if(rule_matches(Bs, As_red) is not None):
        return As_red
    Bs_red = reduce_rule(Bs)
    if(rule_matches(As, Bs_red) is not None):
        return Bs_red
    # if neither sequence matches, there can exist no longer sequence which
    # matches both automata - otherwise we could reduce it
    return None

class TreeParser:
    """ A deterministic bottom-up tree automaton which recognizes precisely
    the language generated by the given TreeGrammar. The constructor will
    throw an exception if the grammar does not yield a deterministic parser
    directly.

    Attributes
    ----------
    _grammar: class TreeGrammar
        The original TreeGrammar.
    _rules: dict
        A dictionary mapping terminal symbols x to tuples (A, [B_1, ..., B_m])
        for each production rules of the form A -> x(B_1, ..., B_m).

    """
    def __init__(self, grammar):
        self._grammar = grammar
        self._rules = {}
        for left in grammar._rules:
            for r in range(len(grammar._rules[left])):
                sym, rights = grammar._rules[left][r]
                # check if the right-hand-side nonterminal sequence is
                # deterministic
                check_rule_determinism(rights)
                # check if a rule with the same right-hand-side symbol
                # already exists
                if(sym not in self._rules):
                    self._rules[sym] = [(left, r, rights)]
                else:
                    sym_rules = self._rules[sym]
                    # check if a rules with the same right hand side
                    # already exists
                    for left2, r2, rights2 in sym_rules:
                        intersect = rules_intersect(rights, rights2)
                        if(intersect is not None):
                            right_strs = [str(right) for right in rights]
                            right_str = ', '.join(right_strs)
                            right_strs = [str(right) for right in rights2]
                            right_str2 = ', '.join(right_strs)
                            raise ValueError('The given grammar was ambiguous: There are two production rules with an intersecting right-hand side, namely %s -> %s(%s) and %s -> %s(%s), both accepting %s' % (left, sym, right_str, left2, sym, right_str2, str(intersect)))
                    # if that is not the case, append the new rule
                    sym_rules.append((left, r, rights))

    def parse_tree(self, x):
        """ The same as parse, but with a tree.Tree object as input.

        Parameters
        ----------
        x: class tree.Tree
            a tree object.

        Returns
        -------
        nont: str
            the nonterminal from which seq generates the input tree.
        seq: list
            a rule sequence seq such that self._grammar.produce_tree(seq, nont)
            is equal to x.

        Raises
        ------
        ValueError
            If the input is not a tree or not part of the language.

        """
        # check if the current node is in the alphabet at all
        if x._label not in self._grammar._alphabet:
            raise ValueError('%s is not part of the alphabet' % str(x._label))
        # then, parse all the children
        actual_rights = []
        seqs = []
        for y in x._children:
            # get the nonterminal and the rule sequence which
            # generates the jth subtree
            nont_y, seq_y = self.parse_tree(y)
            # otherwise, append it to the child nont list
            actual_rights.append(nont_y)
            # and append the rule sequence to the sequence which
            # generates the ith subtree
            seqs.append(seq_y)
        # retrieve the matching production rule for the current situation
        for left, r, rights in self._rules[x._label]:
            match = rule_matches(rights, actual_rights)
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
                                c += 1
                            else:
                                seq.append(0)
                            continue
                        if rights[a].endswith('*'):
                            # if the ath nonterminal is starred, use a 1
                            # production rule for every matched symbol
                            for m in range(len(match[a])):
                                seq.append(1)
                                # then produce the matched child
                                seq += seqs[c]
                                c += 1
                            # finally, use a 0 rule to end the production
                            seq.append(0)
                            continue
                    # in all other cases, just append the production rules
                    # for the current child
                    seq += seqs[c]
                    c += 1
                return left, seq
        # if no such rule exists, the parse fails
        raise ValueError('No rule with %s(%s) on the right-hand side exists' % (str(x._label), str(actual_rights)))
