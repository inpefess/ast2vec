"""
Implements utility functions to parse Python programs as
syntax trees. Also contains a grammar object which defines
the grammar of the Python programming language.

"""

# Copyright (C) 2020
# Benjamin Paaßen, Jessica McBroom
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
__copyright__ = 'Copyright 2020, Benjamin Paaßen, Jessica McBroom'
__license__ = 'GPLv3'
__version__ = '0.1.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'benjamin.paassen@sydney.edu.au'

import ast
from tqdm import tqdm
import tree
import tree_grammar

def ast_to_tree(ast):
    """ Converts a Python abstract syntax tree as returned by the Python
    compiler into a node list/adjacency list format.

    Parameters
    ----------
    ast: class ast.AST
        An abstract syntax tree in Pythons internal compiler format.

    Returns
    -------
    nodes: list
        A node list, where each element is a string identifying a syntactic
        element of the input program (e.g. 'If', 'For', 'expr', etc.).
    adj: list
        An adjacency list defining the tree structure.
    var: dict
        A map of node indices to dictionaries containing variable information for the node
    """
    nodes = []
    adj   = []
    var   = {}
    # go through the AST via depth first search and accumulate nodes,
    # adjacencies, and variables
    _ast_to_tree(ast, nodes, adj, var)
    return nodes, adj, var

def _ast_to_tree(ast_node, nodes, adj, var):
    # append the current node to the tree
    i = len(nodes)
    symbol = ast_node.__class__.__name__
    if symbol == 'ImportFrom':
        # map ImportFrom to Import because they are semantically the same
        symbol = 'Import'
    nodes.append(symbol)
    adj_i = []
    adj.append(adj_i)
    # check if this node has an 'arg' or an 'id' attribute, in which
    # case we need to store a new variable reference
    attributes = ['arg', 'id', 'name', 'attr', 'n', 's', 'is_async', 
                'conversion', 'annotation', 'returns']
    for attribute in attributes:
        if hasattr(ast_node, attribute):
            refdict = var.setdefault(i, {})
            refdict[attribute] = getattr(ast_node, attribute)
    if isinstance(ast_node, ast.NameConstant):
            refdict = var.setdefault(i, {})
            refdict['value'] = getattr(ast_node, 'value')

    if symbol == 'If':
        # explicitly treat if nodes differently because they include
        # two statement lists, which we need to disambiguate
        adj_i.append(len(nodes))
        _ast_to_tree(ast_node.test, nodes, adj, var)
        adj_i.append(len(nodes))
        nodes.append('Then')
        adj_j = []
        adj.append(adj_j)
        for node in ast_node.body:
            adj_j.append(len(nodes))
            _ast_to_tree(node, nodes, adj, var)
        adj_i.append(len(nodes))
        nodes.append('Else')
        adj_j = []
        adj.append(adj_j)
        for node in ast_node.orelse:
            adj_j.append(len(nodes))
            _ast_to_tree(node, nodes, adj, var)
        return
    elif symbol == 'Slice':
        # explicity treat slice nodes to insert intermediate
        # nodes for lower bound, upper bound, and step size
        adj_i.append(len(nodes))
        nodes.append('Lower')
        adj_j = []
        adj.append(adj_j)
        if hasattr(ast_node, 'lower') and ast_node.lower != None:
            adj_j.append(len(nodes))
            _ast_to_tree(ast_node.lower, nodes, adj, var)

        adj_i.append(len(nodes))
        nodes.append('Upper')
        adj_j = []
        adj.append(adj_j)
        if hasattr(ast_node, 'upper') and ast_node.upper != None:
            adj_j.append(len(nodes))
            _ast_to_tree(ast_node.upper, nodes, adj, var)

        adj_i.append(len(nodes))
        nodes.append('Step')
        adj_j = []
        adj.append(adj_j)
        if hasattr(ast_node, 'step') and ast_node.step != None:
            adj_j.append(len(nodes))
            _ast_to_tree(ast_node.step, nodes, adj, var)
        return
    elif(symbol == 'Dict'):
        # explicity treat dict nodes to insert intermediate
        # nodes for key list and value list
        adj_i.append(len(nodes))
        nodes.append('Keys')
        adj_j = []
        adj.append(adj_j)
        for node in ast_node.keys:
            adj_j.append(len(nodes))
            _ast_to_tree(node, nodes, adj, var)
        adj_i.append(len(nodes))
        nodes.append('Values')
        adj_j = []
        adj.append(adj_j)
        for node in ast_node.values:
            adj_j.append(len(nodes))
            _ast_to_tree(node, nodes, adj, var)
        return
    elif(symbol == 'Try'):
        # explicity treat try nodes to insert an intermediate
        # node for finally
        for node in ast_node.body:
            adj_i.append(len(nodes))
            _ast_to_tree(node, nodes, adj, var)
        for node in ast_node.handlers:
            adj_i.append(len(nodes))
            _ast_to_tree(node, nodes, adj, var)
        adj_i.append(len(nodes))
        nodes.append('Finally')
        adj_j = []
        adj.append(adj_j)
        for node in ast_node.finalbody:
            adj_j.append(len(nodes))
            _ast_to_tree(node, nodes, adj, var)
        return
    # handle all children of this node recursively
    for node in ast.iter_child_nodes(ast_node):
        # ignore some nodes
        if(isinstance(node, ast.Load) or isinstance(node, ast.Store) or isinstance(node, ast.Del) or isinstance(node, ast.alias)):
            continue
        adj_i.append(len(nodes))
        _ast_to_tree(node, nodes, adj, var)

def tree_to_ast(nodes, adj, var={}):
    """Converts an abstract syntax tree in node list/adjacency list format to
    an abstract syntax tree in Python compiler format.

    Parameters
    ----------
    nodes: list
        A node list, where each element is a string identifying a syntactic
        element of the input program (e.g. 'If', 'For', 'expr', etc.).
    adj: list
        An adjacency list defining the tree structure.
    var: dict (default = {})
        A map of node indices to dictionaries containing variable information for the node

    Returns
    -------
    ast: class ast.AST
        An abstract syntax tree in Pythons internal compiler format.

    """
    return _tree_to_ast(0, nodes, adj, var)

def _tree_to_ast(node_index, nodes, adj, var):
    symbol = nodes[node_index]
    #print()
    #print(f"processing {symbol}")

    # recursively convert child nodes to trees
    child_indices = adj[node_index]
    child_nodes = [_tree_to_ast(i, nodes, adj, var) for i in child_indices]

    # -------------------------------------------------------------------------------------
    # process any 'special' symbols (i.e. the ones explicitly treated differently in _ast_to_tree)
    if symbol in ["Then", "Else", "Keys", "Values", "Lower", "Upper", "Step", "Finally"] :
        return child_nodes

    if symbol == "If":
        orelse = [] if len(child_nodes) < 3 else child_nodes[2]
        return ast.If(test = child_nodes[0], body = child_nodes[1], orelse = orelse)

    if symbol == "Dict":
        return ast.Dict(keys = child_nodes[0], values = child_nodes[1])

    if symbol == "Try": #NB: not sure what the orelse part is
        
        finalbody = []
        # if there was a finally node, the last child will be in list form
        if isinstance(child_nodes[-1],list):
            finalbody = child_nodes[-1]
            child_nodes = child_nodes[:-1]

        # divide the remaining nodes into the main body + handlers (handlers are of type ExceptHandler)
        body = [x for x in child_nodes if not isinstance(x, ast.ExceptHandler)]
        handlers = [x for x in child_nodes if isinstance(x, ast.ExceptHandler)]

        node = ast.Try(body = body, handlers = handlers, orelse=[], finalbody=finalbody)
        return node

    if symbol == "Import": #the encoding currently throws out the alias info, so make up package name
        return ast.Import(names=[ast.alias(name='something', asname=None)])

    if symbol == 'Slice':
        #these are all lists with 1 element, so we're pulling that element out here
        step = None if child_nodes[2]==[] else child_nodes[2][0]
        lower = None if child_nodes[0] == [] else child_nodes[0][0]
        upper = None if child_nodes[1] == [] else child_nodes[1][0]
        return ast.Slice(lower=lower, upper=upper, step=step)
    # -------------------------------------------------------------------------------------

    # get the fields for this symbol (e.g. body, target, op etc.)
    fields = getattr(ast, symbol)._fields
    #print(f"Just calculated the fields for {symbol}, which are:", fields)

    # if this is a node with no fields (e.g. Eq), return the node immediately
    if len(fields) == 0:
        return getattr(ast, symbol)()

    # set the default field values to be used if var is missing some information
    default_vals = {
        's' :   "<string>",
        'n' :   0,
        'id':   'x',
        'arg':  'keyword',
        'name': 'func',
        'attr': 'attribute',
        'value': True,
        'is_async' : 0, # for loops
        'annotation' : None,
        # I don't really know what this is. I've just put it here so it gets ignored when making a FunctionDef node 
        'decorator_list' : [] ,
        'annotation' : None,
        'returns' : None
    }

    ''' 
    Process any 'standard' nodes. That is, nodes that have the following properties:
    1) they have at most 1 field (excluding keywords) that is a list (e.g. body, args etc.)
    2) they don't have any optional fields
    '''
    standard_symbols = ['Subscript', 'Index', 'Expr','BinOp','Assign', 'BoolOp', 'Call', 'Module','keyword',
                        'Str', 'Num', 'Name', 'Attribute', 'NameConstant', 'List', 'IfExp', 'ListComp',
                        'comprehension', 'DictComp', 'SetComp', 'GeneratorExp', 'Set', 'UnaryOp', 'Tuple',
                        'JoinedStr', 'Lambda', 'arg', 'FunctionDef', 'AugAssign']
    if symbol in standard_symbols:

        # only keep the value field if this is a NameConstant
        if symbol != 'NameConstant':
            del default_vals['value']

        # remove any ignored fields
        ignored_fields = ['ctx']
        fields = [x for x in fields if x not in ignored_fields]
        
        # get the values of any fields from var (if var is missing anything, use a default value)
        arguments_dict = var[node_index] if node_index in var else {}
        arguments_dict = {x : arguments_dict[x] for x in arguments_dict if x in fields} # remove any invalid arguments
        for field in fields:
            if field not in arguments_dict and field in default_vals:
                arguments_dict[field] = default_vals[field]
        fields = [x for x in fields if x not in default_vals]

        # add any keyword arguments to the arguments dict
        if 'keywords' in fields:
            # assign the keyword nodes to the keywords field
            arguments_dict['keywords'] = [x for x in child_nodes if isinstance(x, ast.keyword)]

            # remove keywords from children + fields
            child_nodes = [x for x in child_nodes if not isinstance(x, ast.keyword)]
            fields.remove('keywords')

        '''
        Process the remaining fields
        starting from the left, assign children to fields on a 1-to-1 basis 
        (i.e. child 1 to field 1, child 2 to field 2) until we reach a field that's a list (like body)
        then, do the same thing from the right (i.e. child -1 to field -1, child -2 to field -2)
        If there are any unassigned children at the end, they will belong to that list field, so assign them there
        '''
        list_fields = ['body', 'targets', 'values','args','elts', 'generators','ifs']
        if symbol in ['IfExp', 'Lambda']: #in these cases, body holds a single node
            list_fields.remove('body')
        if symbol in ['Lambda', 'FunctionDef']:
            list_fields.remove('args')

        # assign from the left
        while len(child_nodes) > 0:
            if fields[0] in list_fields:
                break
            arguments_dict[fields[0]] = child_nodes[0]
            child_nodes.pop(0)
            fields.pop(0)
        # assign from the right
        while len(child_nodes) > 0:
            if fields[-1] in list_fields:
                break
            arguments_dict[fields[-1]] = child_nodes[-1]
            child_nodes.pop(-1)
            fields.pop(-1)
        # assign all remaining children to the last field (if required)
        if len(fields) > 0:
            arguments_dict[fields[0]] = child_nodes

        #print("returning:")
        result = getattr(ast, symbol)(** arguments_dict)
        #print(ast.dump(result))
        return result


    # process other symbols manually
    if symbol == 'Compare':
        # calculate the number of operators from the number of children
        # e.g. there would be 5 child nodes for 1 < a < 10, which would mean 2 operators
        num_operators = int((len(child_nodes) - 1)/2)
        ops = child_nodes[1:num_operators + 1]
        comparators = child_nodes[num_operators + 1:]
        return ast.Compare(left = child_nodes[0], ops = ops, comparators = comparators)

    if symbol == 'ExceptHandler': 
        arguments_dict = {}

        # decide if the optional field, "type", has a value based on the type of the first child
        if isinstance(child_nodes[0], ast.Name):
            arguments_dict['type'] = child_nodes[0]
            child_nodes.pop(0)
        else:
            arguments_dict['type'] = None
        arguments_dict['body'] = child_nodes
        # get the exception name from var if it's there
        arguments_dict['name'] = None
        if node_index in var:
            arguments_dict['name'] = var[node_index].get("name", None)
        
        return ast.ExceptHandler(** arguments_dict)

    '''
    NB: this doesn't deal properly with loops with else statements 
    e.g.
    for i in range(10):
        .... stuff
    else:
        .... other stuff
    (everything in the else statement will end up in the main body of the loop)
    (probably not possible to disambiguate from current tree?)
    '''
    if symbol == "For": 
        return ast.For(target = child_nodes[0], iter=child_nodes[1],
                        body=child_nodes[2:], orelse=[])
    if symbol == "While": 
        return ast.While(test = child_nodes[0], body=child_nodes[1:], orelse=[])

    if symbol == "FormattedValue":
        format_spec = child_nodes[1] if len(child_nodes) == 2 else None
        conversion = -1 #-1 is default no formatting
        if node_index in var:
            conversion = var[node_index].get('conversion', -1) 
        return  ast.FormattedValue(value = child_nodes[0], conversion = conversion, format_spec = format_spec)

    '''
    NB: this doesn't deal with arbitrary arguments or keyword arguments
    all arguments are assumed to be the default kind
    (don't think it's possible to distiguish the others from the default kind?)
    e.g. something like 'def x(z, *a, b, x=7):\n  pass' will just end up with def x(z, a, b, x):
    '''
    if symbol == "arguments":
        return ast.arguments(args = child_nodes, vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])



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
        A list of triples, each of which contains a node list as first,
        an adjacency list as second, and a dictionary of variable names
        mapped to node indices as third argument. Note that this list may
        be shorter than the input source code list if filter_uniques is
        set to True.
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
            nodes, adj, var = ast_to_tree(ast.parse(srcs[i], 'program.py'))
            # ensure that the tree conforms to the python grammar
            parser.parse(nodes, adj)
            # only append it to the tree list if it is compileable _and_
            # parseable
            trees.append((nodes, adj, var))
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
            nodes, adj, var = trees[i]
            is_unique = True

            '''
            Old version:
            for nodes2, adj2, var2 in trees_filtered:
                if nodes == nodes2 and adj == adj2:
                    is_unique = False
                    break
            '''
            for j, tree in enumerate(trees_filtered): #jess edit
                nodes2, adj2, var2 = tree #jess edit
                if nodes == nodes2 and adj == adj2:
                    is_unique = False
                    id_map[tree_ids[i]] = j  #jess edit 
                    break
            if is_unique:
                id_map[tree_ids[i]] = len(trees_filtered)
                trees_filtered.append((nodes, adj, var))
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
    'Num' : 0, 'Str' : 0, 'NameConstant' : 0, 'Attribute' : 1, 'Bytes' : 0, 'Ellipsis' : 0,
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
    'except', 'comprehension', 'keys_jess', 'values_jess',
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
        ('Dict', ['keys_jess', 'values_jess']),
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
        ('Num', []), ('Str', []), ('NameConstant', []), ('Bytes', []), ('Ellipsis', []),
        ('Attribute', ['expr']),
        ('Subscript', ['expr', 'slice']),
        ('Starred', ['expr']),
        ('Name', []), ('List', ['expr*']), ('Tuple', ['expr*'])
    ],
    'keys_jess' : [('Keys', ['expr*'])],
    'values_jess' : [('Values', ['expr*'])],
    'slice' : [
        ('Index', ['expr']),
        ('Slice', ['lower', 'upper', 'step']),
        ('ExtSlice', ['slice*'])
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
