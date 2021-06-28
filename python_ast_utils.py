"""
Implements utility functions to parse Python programs as
syntax trees. Also contains a grammar object which defines
the grammar of the Python programming language.

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

import ast
from tqdm import tqdm
import tree
import tree_grammar

def ast_to_tree(ast_node):
    """ Converts a Python abstract syntax tree as returned by the Python
    compiler into a tree.Tree

    Parameters
    ----------
    ast_node: class ast.AST
        An abstract syntax tree in Pythons internal compiler format.

    Returns
    -------
    x: class tree.Tree
        A tree object according to the tree.Tree class. At each node, we
        additionally annotate variable references.

    """

    label = ast_node.__class__.__name__
    if label == 'ImportFrom':
        # map ImportFrom to Import because they are semantically the same
        label = 'Import'

    # set up an output tree node
    out_node = tree.Tree(label)

    # check for certain attributes and store them at the node
    attributes = ['arg', 'id', 'name', 'attr', 'is_async', 
                'conversion', 'annotation', 'returns']

    for attribute in attributes:
        if hasattr(ast_node, attribute):
            setattr(out_node, attribute, getattr(ast_node, attribute))
    if label == 'Constant':
        out_node.value = ast_node.value

    # consider a few special cases where we need to convert to conform to the
    # grammar
    if label == 'If':
        # add test as first child
        out_node._children.append(ast_to_tree(ast_node.test))
        # disambiguate the then and the else block in an if statement
        # by using an intermediate node
        then_node = tree.Tree('Then')
        out_node._children.append(then_node)
        for node in ast_node.body:
            then_node._children.append(ast_to_tree(node))

        else_node = tree.Tree('Else')
        out_node._children.append(else_node)
        for node in ast_node.orelse:
            else_node._children.append(ast_to_tree(node))
        return out_node
    elif label == 'Slice':
        # explicity treat slice nodes to insert intermediate
        # nodes for lower bound, upper bound, and step size
        lower_node = tree.Tree('Lower')
        if hasattr(ast_node, 'lower') and ast_node.lower != None:
            lower_node._children.append(ast_to_tree(ast_node.lower))

        upper_node = tree.Tree('Upper')
        if hasattr(ast_node, 'upper') and ast_node.upper != None:
            upper_node._children.append(ast_to_tree(ast_node.upper))

        step_node = tree.Tree('Step')
        if hasattr(ast_node, 'step') and ast_node.step != None:
            step_node._children.append(ast_to_tree(ast_node.step))

        out_node._children = [lower_node, upper_node, step_node]
        return out_node
    elif label == 'Subscript':
        # add the array that is subscripted as first child
        out_node._children.append(ast_to_tree(ast_node.value))
        # then handle the case where the slice is not an explicit Slice node
        if not isinstance(ast_node.slice, ast.Slice):
            index_node = tree.Tree('Index')
            index_node._children.append(ast_to_tree(ast_node.slice))
            out_node._children.append(index_node)
        else:
            out_node._children.append(ast_to_tree(ast_node.slice))
        return out_node
    elif label == 'Dict':
        # explicity treat dict nodes to insert intermediate
        # nodes for key list and value list
        keys_node = tree.Tree('Keys')
        for node in ast_node.keys:
            keys_node._children.append(ast_to_tree(node))

        values_node = tree.Tree('Values')
        for node in ast_node.values:
            values_node._children.append(ast_to_tree(node))

        out_node._children = [keys_node, values_node]

        return out_node
    elif label == 'Try':
        # explicity treat try nodes to insert an intermediate
        # node for finally
        for node in ast_node.body:
            out_node._children.append(ast_to_tree(node))
        for node in ast_node.handlers:
            out_node._children.append(ast_to_tree(node))
        finally_node = tree.Tree('Finally')
        out_node._children.append(finally_node)
        for node in ast_node.finalbody:
            finally_node._children.append(ast_to_tree(node))
        return out_node
    else:
        # if none of the special cases applied, handle the children recursively.
        # handle all children of this node recursively
        for node in ast.iter_child_nodes(ast_node):
            # ignore some nodes
            if isinstance(node, ast.Load) or isinstance(node, ast.Store) or isinstance(node, ast.Del) or isinstance(node, ast.alias):
                continue
            out_node._children.append(ast_to_tree(node))
        return out_node


def tree_to_ast(tree):
    """ Converts a tree back into a Python internal AST object, plugging in
    default values for variable names and constants if necessary.

    Parameters
    ----------
    tree: class tree.Tree
        A tree.Tree object representing a Python syntax tree.

    Returns
    -------
    ast_node: class ast.AST
        The same tree in Python's internal AST format.

    """
    # recursively convert the children
    child_asts = [tree_to_ast(child) for child in tree._children]

    # certain nodes need to be ignored during conversion
    # because they are merely auxiliary nodes for our
    # AST format
    if tree._label in ['Then', 'Else', 'Keys', 'Values', 'Lower', 'Upper', 'Step', 'Index', 'Finally']:
        return child_asts

    # handle nodes which do not require any children or further
    # information
    if tree._label in _trivial_nodes:
        return getattr(ast, tree._label)()

    # handle module nodes
    if tree._label == 'Module':
        return ast.Module(body = child_asts)

    # handle statement nodes
    if tree._label == 'FunctionDef':
        if not hasattr(tree, 'name'):
            name = 'f'
        else:
            name = tree.name
        return ast.FunctionDef(name = name, args = child_asts[0], body = child_asts[1:], decorator_list = [])

    if tree._label == 'ClassDef':
        if not hasattr(tree, 'name'):
            name = 'C'
        else:
            name = tree.name
        # collect bases. These must be expression nodes
        i = 0
        while i < len(child_asts) and tree._children[i]._label in _expr_types:
            i += 1
        return ast.ClassDef(name = name, bases = child_asts[:i], body = child_asts[i:])

    if tree._label == 'Return':
        val = child_asts[0] if len(child_asts) > 0 else None
        return ast.Return(value = val)

    if tree._label == 'Delete':
        return ast.Delete(targets = child_asts)

    if tree._label == 'Assign':
        return ast.Assign(targets = child_asts[:-1], value = child_asts[-1])

    if tree._label == 'AugAssign':
        return ast.AugAssign(target = child_asts[0], op = child_asts[1], value = child_asts[2])

    if tree._label == 'AnnAssign':
        val = child_asts[2] if len(child_asts) >= 3 else None
        return ast.AnnAssign(target = child_asts[0], annotation = child_asts[1], value = val)

    if tree._label == 'For':
        return ast.For(target = child_asts[0], iter = child_asts[1], body = child_asts[2:], orelse = [])

    if tree._label == 'While':
        return ast.For(target = child_asts[0], body = child_asts[1:])

    if tree._label == 'If':
        orelse = [] if len(child_asts) < 3 else child_asts[2]
        return ast.If(test = child_asts[0], body = child_asts[1], orelse = orelse)

    if tree._label == 'With':
        # collect all withitems
        i = 0
        while i < len(tree._children) and tree._children[i]._label == 'withitem':
            i += 1
        return ast.With(items = child_asts[:i], body = child_asts[i:])

    if tree._label == 'Raise':
        return ast.Raise(exc = child_asts[0])

    if tree._label == 'Try':
        # check if there are exception handlers
        i = len(tree._children)-1
        while i > 0 and tree._children[i-1]._label == 'ExceptHandler':
            i -= 1
        return ast.Try(body = child_asts[:i], handlers = child_asts[i:-1], finalbody = child_asts[-1])

    if tree._label == 'Assert':
        msg = child_asts[1] if len(child_asts) > 1 else None
        return ast.Assert(test = child_asts[0], msg = msg)

    if tree._label == 'Import':
        return ast.Import(names=[ast.alias(name = 'some_package', asname = None)])

    if tree._label == 'Expr':
        return ast.Expr(value = child_asts[0])

    # handle expression nodes
    if tree._label == 'BoolOp':
        return ast.BoolOp(op = child_asts[0], values = child_asts[1])

    if tree._label == 'BinOp':
        return ast.BinOp(left = child_asts[0], op = child_asts[1], right = child_asts[2])

    if tree._label == 'UnaryOp':
        return ast.UnaryOp(op = child_asts[0], operand = child_asts[1])

    if tree._label == 'Lambda':
        return ast.Lambda(args = child_asts[0], body = child_asts[1])

    if tree._label == 'IfExp':
        return ast.IfExp(test = child_asts[0], body = child_asts[1], orelse = child_asts[2])

    if tree._label == 'Dict':
        return ast.Dict(keys = child_asts[0], values = child_asts[1])

    if tree._label == 'Set':
        return ast.Set(elts = child_asts)

    if tree._label == 'ListComp':
        return ast.ListComp(elt = child_asts[0], generators = child_asts[1:])

    if tree._label == 'SetComp':
        return ast.SetComp(elt = child_asts[0], generators = child_asts[1:])

    if tree._label == 'DictComp':
        return ast.DictComp(key = child_asts[0], value = child_asts[1], generators = child_asts[2:])

    if tree._label == 'GeneratorExp':
        return ast.GeneratorExp(elt = child_asts[0], generators = child_asts[1:])

    if tree._label == 'Yield':
        val = child_asts[0] if len(child_asts) > 0 else None
        return ast.Yield(value = val)

    if tree._label == 'Compare':
        # retrieve all operators
        i = 1
        while i < len(tree._children) and tree._children[i]._label in _cmp_ops:
            i += 1
        return ast.Compare(left = child_asts[0], ops = child_asts[1:i], comparators = child_asts[i:])

    if tree._label == 'Call':
        # retrieve all keywords
        i = len(tree._children)
        while i > 0 and tree._children[i-1]._label == 'keyword':
            i -= 1
        return ast.Call(func = child_asts[0], args = child_asts[1:i], keywords = child_asts[i:])

    if tree._label == 'FormattedValue':
        conversion = tree.conversion if hasattr(tree, 'conversion') else -1
        format_spec = child_asts[1] if len(child_asts) > 1 else None
        return ast.FormattedValue(value = child_asts[0], conversion = conversion, format_spec = format_spec)

    if tree._label == 'JoinedStr':
        return ast.JoinedStr(values = child_asts)

    if tree._label == 'Constant':
        if not hasattr(tree, 'value'):
            val = '<some_value>'
        else:
            val = tree.value
        # for this, we copy the value field
        return ast.Constant(value = val)

    if tree._label == 'Attribute':
        if hasattr(tree, 'attr'):
            attr = tree.attr
        else:
            attr = 'attr'
        return ast.Attribute(value = child_asts[0], attr = attr)

    if tree._label == 'Subscript':
        return ast.Subscript(value = child_asts[0], slice = child_asts[1])

    if tree._label == 'Starred':
        return ast.Starred(value = child_asts[0])

    if tree._label == 'Name':
        if hasattr(tree, 'id'):
            identifier = tree.id
        else:
            identifier = 'x'
        return ast.Name(id = identifier)

    if tree._label == 'List':
        return ast.List(elts = child_asts)

    if tree._label == 'Tuple':
        return ast.Tuple(elts = child_asts)

    if tree._label == 'Slice':
        lower = child_asts[0][0] if len(child_asts[0]) > 0 else None
        upper = child_asts[1][0] if len(child_asts[1]) > 0 else None
        step  = child_asts[2][0] if len(child_asts[2]) > 0 else None
        return ast.Slice(lower=lower, upper=upper, step=step)

    if tree._label == 'comprehension':
        return ast.comprehension(target = child_asts[0], iter = child_asts[1], ifs = child_asts[2:])

    if tree._label == 'ExceptHandler':
        if len(tree._children) > 0 and tree._children[0]._label in _expr_types:
            tp = child_asts[0]
            body = child_asts[1:]
        else:
            tp = None
            body = child_asts
        return ast.ExceptHandler(type = tp, body = body)

    if tree._label == 'arguments':
        # retrieve all arguments first
        i = 0
        while i < len(tree._children) and tree._children[i]._label == 'arg':
            i += 1
        return ast.arguments(args = child_asts[:i], defaults = child_asts[i:])

    if tree._label == 'arg':
        arg = tree.arg if hasattr(tree, 'arg') else 'x'
        return ast.arg(arg = arg)

    if tree._label == 'keyword':
        arg = tree.arg if hasattr(tree, 'arg') else None
        return ast.keyword(arg = arg, value = child_asts[0])

    if tree._label == 'withitem':
        optional_vars = child_asts[1] if len(child_asts) > 1 else None
        return ast.withitem(context_expr = child_asts[0], optional_vars = child_asts[1])

    raise ValueError('Unexpected tree label: %s' % str(tree._label))



def parse_asts(srcs, filter_uniques = False, ids = None, ignore_syntax_errors = False, verbose = False):
    """ Parse an abstract syntax tree from Python code.

    Parameters
    ----------
    srcs: list
        A list of Python source code strings.
    filter_uniques: bool (default = False)
        If set to True, the output list may be shorter than the srcs input
        list because only unique trees are kept.
    ids: list (default = None)
        A list of hashable ids for every source code string.
    ignore_syntax_errors: bool (default = False)
        If set to true, non-compiling programs will be ignored.
    verbose: bool (default = False)
        If set to true, a progress bar is printed using tqdm.

    Returns
    -------
    trees: list
        A list of tree.Tree objects, each representing a syntax tree.
        Note that nodes have additional properties to store auxiliary
        information, like variable names.
        Note that this list may be shorter than the input source code list
        if filter_uniques is set to True.
    id_map: dict
        A dictionary of indices in the src list to indices in the trees
        list OR a dictionary of ids to indices in the tree list, if
        ids is not None. This second output is only returned if
        filter_uniques is set to True.

    """
    if not verbose:
        progbar = lambda x : x
    else:
        progbar = tqdm
        print('starting parsing')

    if ids is None:
        ids = list(range(len(srcs)))

    parser = tree_grammar.TreeParser(grammar)

    trees = []
    tree_ids = []
    num_errors = 0
    for i in progbar(range(len(srcs))):
        try:
            x = ast_to_tree(ast.parse(srcs[i], 'program.py'))
            # ensure that the tree conforms to the python grammar
            parser.parse_tree(x)
            # only append it to the tree list if it is compileable _and_
            # parseable
            trees.append(x)
        except (SyntaxError, ValueError, AttributeError) as ex:
            if ignore_syntax_errors:
                num_errors += 1
                continue
            else:
                raise ex
        tree_ids.append(ids[i])

    if ignore_syntax_errors and verbose:
        print('Warning: %d of %d programs did not compile due to syntax errors.' % (num_errors, len(srcs)))

    # filter the tree list
    if filter_uniques:
        if verbose:
            print('starting filtering')
        id_map = {}
        trees_filtered = []
        for i in progbar(range(len(trees))):
            x = trees[i]
            is_unique = True
            for j, y in enumerate(trees_filtered):
                if x == y:
                    is_unique = False
                    id_map[tree_ids[i]] = j
                    break
            if is_unique:
                id_map[tree_ids[i]] = len(trees_filtered)
                trees_filtered.append(x)
        return trees_filtered, id_map
    else:
        return trees

# This is a list of builtin functions in the Python language which are
# all available without imports
_builtin_funs = ['abs', 'delattr', 'hash', 'memoryview', 'set', 'all', 'dict', 'help', 'min', 'setattr', 'any', 'dir', 'hex', 'next', 'slice', 'ascii', 'divmod', 'id', 'object', 'sorted', 'bin', 'enumerate', 'input', 'oct', 'staticmethod', 'bool', 'eval', 'int', 'open', 'str', 'breakpoint', 'exec', 'isinstance', 'ord', 'sum', 'bytearray', 'filter', 'issubclass', 'pow', 'super', 'bytes', 'float', 'iter', 'print', 'tuple', 'callable', 'format', 'len', 'property', 'type', 'chr', 'frozenset', 'list', 'range', 'vars', 'classmethod', 'getattr', 'locals', 'repr', 'zip', 'compile', 'globals', 'map', 'reversed', '__import__', 'complex', 'hasattr', 'max', 'round']


# this AST grammar is taken pretty much directly from
# https://docs.python.org/3/library/ast.html
_alphabet = { 'Module' : 1,
    # statement nodes
    'FunctionDef' : 2, 'ClassDef' : 2, 'Return' : 1, 'Delete' : 1,
    'Assign' : 2, 'AugAssign' : 3, 'AnnAssign' : 3,
    'For' : 3, 'While' : 2, 'If' : 3,
    'With' : 2, 'Raise' : 1, 'Try' : 3, 'Assert' : 2,
    'Import' : 0, 'Global' : 0, 'Expr' : 1,
    'Pass' : 0, 'Break' : 0, 'Continue' : 0,
    # Artificial constructs for disambiguation
    'Then' : 1, 'Else' : 1,
    # expression nodes
    'BoolOp' : 2, 'BinOp' : 3, 'UnaryOp' : 2, 'Lambda' : 2, 'IfExp' : 3,
    'Dict' : 2, 'Set' : 1,
    'ListComp' : 2, 'SetComp' : 2, 'DictComp' : 3, 'GeneratorExp' : 2,
    'Yield' : 1, 'Compare' : 3, 'Call' : 3,
    'FormattedValue' : 2, 'JoinedStr' : 1,
    'Constant' : 0, 'Attribute' : 1,
    'Subscript' : 2, 'Starred' : 1, 'Name' : 0, 'List' : 1, 'Tuple' : 1,
    # slice operators
    'Index' : 1, 'Slice' : 3, 'Lower' : 1, 'Upper' : 1, 'Step' : 1, 'ExtSlice' : 1,
    # boolean operators
    'And' : 0, 'Or' : 0,
    # binary operators
    'Add' : 0, 'Sub' : 0, 'Mult' : 0, 'MatMult' : 0, 'Div' : 0, 'Mod' : 0,
    'Pow' : 0, 'LShift' : 0, 'RShift' : 0, 'BitOr' : 0, 'BitXor' : 0, 'BitAnd' : 0, 'FloorDiv' : 0,
    # unary operators
    'Invert' : 0, 'Not' : 0, 'UAdd' : 0, 'USub' : 0,
    # comparison operators
    'Eq' : 0, 'NotEq' : 0, 'Lt' : 0, 'LtE' : 0, 'Gt' : 0, 'GtE' : 0,
    'Is' : 0, 'IsNot' : 0, 'In' : 0, 'NotIn' : 0,
    # arguments
    'arguments' : 2, 'arg': 0,
    # exception handling
    'ExceptHandler' : 2,
    'Finally' : 1,
    # list comprehensions
    'comprehension' : 3,
    # dictionary concepts
    'Keys' : 1, 'Values' : 1,
    # with
    'withitem' : 2,
    # keywords
    'keyword' : 1
}
_nonterminals = [ 'mod', 'stmt', 'then', 'else', 'expr', 'slice',
    'lower', 'upper', 'step',
    'boolop', 'operator', 'unaryop', 'cmpop', 'arguments', 'arg',
    'except', 'comprehension', 'dict_keys', 'dict_values',
    'withitem', 'finally', 'keyword']
_start = 'mod'
_rules = {
    'mod'  : [('Module', ['stmt*'])],
    'stmt' : [
        ('FunctionDef', ['arguments', 'stmt*']),
        ('ClassDef', ['expr*', 'stmt*']),
        ('Return', ['expr?']),
        ('Delete', ['expr*']),
        ('Assign', ['expr*', 'expr']),
        ('AugAssign', ['expr', 'operator', 'expr']),
        ('AnnAssign', ['expr', 'expr', 'expr?']),
        ('For', ['expr', 'expr', 'stmt*']),
        ('While', ['expr', 'stmt*']),
        ('If', ['expr', 'then', 'else']),
        ('With', ['withitem*', 'stmt*']),
        ('Raise', ['expr']),
        ('Try', ['stmt*', 'except*', 'finally']),
        ('Assert', ['expr', 'expr?']),
        ('Import', []),
        ('Global', []),
        ('Expr', ['expr']),
        ('Pass', []), ('Break', []), ('Continue', []),
    ],
    'then' : [('Then', ['stmt*'])],
    'else' : [('Else', ['stmt*'])],
    'expr' : [
        ('BoolOp', ['boolop', 'expr*']),
        ('BinOp', ['expr', 'operator', 'expr']),
        ('UnaryOp', ['unaryop', 'expr']),
        ('Lambda', ['arguments', 'expr']),
        ('IfExp', ['expr', 'expr', 'expr']),
        ('Dict', ['dict_keys', 'dict_values']),
        ('Set', ['expr*']),
        ('ListComp', ['expr', 'comprehension*']),
        ('SetComp', ['expr', 'comprehension*']),
        ('DictComp', ['expr', 'expr', 'comprehension*']),
        ('GeneratorExp', ['expr', 'comprehension*']),
        ('Yield', ['expr?']),
        ('Compare', ['expr', 'cmpop*', 'expr*']),
        ('Call', ['expr', 'expr*', 'keyword*']),
        ('FormattedValue', ['expr', 'expr?']),
        ('JoinedStr', ['expr*']),
        ('Constant', []), # changed in Python 3.8
        ('Attribute', ['expr']),
        ('Subscript', ['expr', 'slice']),
        ('Starred', ['expr']),
        ('Name', []), ('List', ['expr*']), ('Tuple', ['expr*'])
    ],
    'dict_keys' : [('Keys', ['expr*'])],
    'dict_values' : [('Values', ['expr*'])],
    'slice' : [
        ('Index', ['expr']),
        ('Slice', ['lower', 'upper', 'step'])
    ],
    'lower' : [('Lower', ['expr?'])],
    'upper' : [('Upper', ['expr?'])],
    'step' :  [('Step', ['expr?'])],
    'boolop' : [('And', []), ('Or', [])],
    'operator' : [
        ('Add', []), ('Sub', []), ('Mult', []), ('MatMult', []),
        ('Div', []), ('Mod', []), ('Pow', []), ('LShift', []), ('RShift', []),
        ('BitOr', []), ('BitXor', []), ('BitAnd', []), ('FloorDiv', [])
    ],
    'unaryop' : [('Invert', []), ('Not', []), ('UAdd', []), ('USub', [])],
    'cmpop' : [('Eq', []), ('NotEq', []), ('Lt', []), ('LtE', []),
        ('Gt', []), ('GtE', []), ('Is', []), ('IsNot', []),
        ('In', []), ('NotIn', [])],
    'comprehension' : [('comprehension', ['expr', 'expr', 'expr*'])],
    'except' : [('ExceptHandler', ['expr?', 'stmt*'])],
    'finally' : [('Finally', ['stmt*'])],
    'arguments' : [('arguments', ['arg*', 'expr*'])],
    'arg' : [('arg', [])],
    'keyword' : [('keyword', ['expr'])],
    'withitem' : [('withitem', ['expr', 'expr?'])]
}

grammar = tree_grammar.TreeGrammar(_alphabet, _nonterminals, _start, _rules)


_expr_types = [rhs[0] for rhs in _rules['expr']]
_cmp_ops = [rhs[0] for rhs in _rules['cmpop']]
_trivial_nodes = ['Global', 'Pass', 'Break', 'Continue'] + [rhs[0] for rhs in _rules['boolop']] + [rhs[0] for rhs in _rules['operator']] + [rhs[0] for rhs in _rules['unaryop']] + _cmp_ops
